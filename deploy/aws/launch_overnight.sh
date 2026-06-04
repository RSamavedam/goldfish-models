#!/usr/bin/env bash
# Provision a single EC2 instance for an overnight goldfish-models sweep.
#
# Why this exists: the CDK deploy was hitting bootstrap-ordering bugs.
# This script does the bare minimum to get a usable instance — plain
# Amazon Linux 2023, SSH-accessible, with an IAM role that can read SSM
# secrets + write to S3. You then SSH in and run the sweep by hand.
#
# Idempotent: re-running picks up existing resources by name rather than
# erroring. Safe to re-run if it fails partway through.
#
# Run from any directory:
#   bash deploy/aws/launch_overnight.sh
#
# Final output is the SSH command you need.

set -euo pipefail

# --------------------------------------------------------------------- #
# Config — change these if you want different sizing / region           #
# --------------------------------------------------------------------- #
REGION="${AWS_REGION:-us-east-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-c7i.4xlarge}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-250}"
KEY_NAME="${KEY_NAME:-goldfish-overnight}"
SG_NAME="${SG_NAME:-goldfish-overnight}"
ROLE_NAME="${ROLE_NAME:-goldfish-overnight-role}"
PROFILE_NAME="${PROFILE_NAME:-goldfish-overnight-profile}"
INSTANCE_NAME="${INSTANCE_NAME:-goldfish-overnight}"
KEY_PATH="${KEY_PATH:-${HOME}/${KEY_NAME}.pem}"

export AWS_REGION="${REGION}"
export AWS_DEFAULT_REGION="${REGION}"

# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #
say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m!!!\033[0m %s\n' "$*" >&2; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    err "required command not found: $1"
    exit 1
  }
}
require_cmd aws
require_cmd curl

# --------------------------------------------------------------------- #
# Pre-flight                                                            #
# --------------------------------------------------------------------- #
say "AWS identity:"
aws sts get-caller-identity --output text --query 'Arn'
say "Region: ${REGION}"
echo

# --------------------------------------------------------------------- #
# Subnet                                                                #
# --------------------------------------------------------------------- #
say "Finding a default-VPC public subnet…"
SUBNET_ID=$(aws ec2 describe-subnets \
  --filters "Name=default-for-az,Values=true" \
  --query "Subnets[0].SubnetId" \
  --output text)
[[ "${SUBNET_ID}" == "None" || -z "${SUBNET_ID}" ]] && {
  err "no default-VPC subnet found in ${REGION}"
  exit 1
}
echo "    Subnet: ${SUBNET_ID}"

# --------------------------------------------------------------------- #
# AMI                                                                   #
# --------------------------------------------------------------------- #
say "Resolving latest Amazon Linux 2023 AMI…"
AMI_ID=$(aws ssm get-parameter \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-6.1-x86_64 \
  --query "Parameter.Value" --output text)
echo "    AMI: ${AMI_ID}"

# --------------------------------------------------------------------- #
# Security group                                                        #
# --------------------------------------------------------------------- #
MY_IP=$(curl -s https://checkip.amazonaws.com)
say "Security group ${SG_NAME} (ssh from ${MY_IP}/32)…"
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${SG_NAME}" \
  --query "SecurityGroups[0].GroupId" --output text 2>/dev/null || true)
if [[ "${SG_ID}" == "None" || -z "${SG_ID}" ]]; then
  SG_ID=$(aws ec2 create-security-group \
    --group-name "${SG_NAME}" \
    --description "ssh from raghav laptop for goldfish-models overnight sweep" \
    --query "GroupId" --output text)
  echo "    Created SG: ${SG_ID}"
else
  echo "    Reusing SG:  ${SG_ID}"
fi
# Add ingress for current IP if not already there.
if ! aws ec2 describe-security-groups --group-ids "${SG_ID}" \
      --query "SecurityGroups[0].IpPermissions[?ToPort==\`22\`].IpRanges[].CidrIp" \
      --output text | tr '\t' '\n' | grep -qx "${MY_IP}/32"; then
  aws ec2 authorize-security-group-ingress \
    --group-id "${SG_ID}" \
    --protocol tcp --port 22 --cidr "${MY_IP}/32" >/dev/null
  echo "    Added ingress for ${MY_IP}/32"
else
  echo "    Ingress for ${MY_IP}/32 already present"
fi

# --------------------------------------------------------------------- #
# SSH key                                                               #
# --------------------------------------------------------------------- #
say "SSH keypair ${KEY_NAME}…"
if aws ec2 describe-key-pairs --key-names "${KEY_NAME}" >/dev/null 2>&1; then
  if [[ ! -f "${KEY_PATH}" ]]; then
    err "key ${KEY_NAME} exists in AWS but the private key file ${KEY_PATH} is missing."
    err "You can't recover the private key — AWS only returns it at create time."
    err "Either delete the AWS key (\`aws ec2 delete-key-pair --key-name ${KEY_NAME}\`) so this"
    err "script can recreate it, or set KEY_NAME=<something-else> and re-run."
    exit 1
  fi
  echo "    Reusing existing key (private key: ${KEY_PATH})"
else
  aws ec2 create-key-pair --key-name "${KEY_NAME}" \
    --query "KeyMaterial" --output text > "${KEY_PATH}"
  chmod 600 "${KEY_PATH}"
  echo "    Created key (private key: ${KEY_PATH})"
fi

# --------------------------------------------------------------------- #
# IAM role + instance profile                                           #
# --------------------------------------------------------------------- #
say "IAM role ${ROLE_NAME}…"
if aws iam get-role --role-name "${ROLE_NAME}" >/dev/null 2>&1; then
  echo "    Reusing role"
else
  aws iam create-role --role-name "${ROLE_NAME}" \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
    >/dev/null
  echo "    Created role"
fi

for arn in \
  arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess \
  arn:aws:iam::aws:policy/AmazonS3FullAccess \
  arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
do
  if aws iam list-attached-role-policies --role-name "${ROLE_NAME}" \
      --query "AttachedPolicies[?PolicyArn==\`${arn}\`]" --output text | grep -q .; then
    :  # already attached
  else
    aws iam attach-role-policy --role-name "${ROLE_NAME}" --policy-arn "${arn}"
    echo "    Attached ${arn##*/}"
  fi
done

say "Instance profile ${PROFILE_NAME}…"
if aws iam get-instance-profile --instance-profile-name "${PROFILE_NAME}" >/dev/null 2>&1; then
  echo "    Reusing profile"
else
  aws iam create-instance-profile --instance-profile-name "${PROFILE_NAME}" >/dev/null
  aws iam add-role-to-instance-profile \
    --instance-profile-name "${PROFILE_NAME}" \
    --role-name "${ROLE_NAME}"
  echo "    Created profile"
fi

# IAM is eventually consistent — give role/profile association time to propagate
# before RunInstances tries to attach it.
say "Waiting 15s for IAM to propagate…"
sleep 15

# --------------------------------------------------------------------- #
# Reuse-or-create the EC2 instance                                       #
# --------------------------------------------------------------------- #
say "Checking for an existing running instance named ${INSTANCE_NAME}…"
EXISTING=$(aws ec2 describe-instances \
  --filters \
    "Name=tag:Name,Values=${INSTANCE_NAME}" \
    "Name=instance-state-name,Values=pending,running" \
  --query "Reservations[].Instances[].InstanceId" \
  --output text)
if [[ -n "${EXISTING}" && "${EXISTING}" != "None" ]]; then
  INSTANCE_ID="${EXISTING%%[[:space:]]*}"
  echo "    Reusing existing instance ${INSTANCE_ID}"
else
  say "Launching new ${INSTANCE_TYPE} instance…"
  INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "${AMI_ID}" \
    --instance-type "${INSTANCE_TYPE}" \
    --key-name "${KEY_NAME}" \
    --security-group-ids "${SG_ID}" \
    --subnet-id "${SUBNET_ID}" \
    --iam-instance-profile "Name=${PROFILE_NAME}" \
    --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${VOLUME_SIZE_GB},VolumeType=gp3,Encrypted=true,DeleteOnTermination=true}" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}},{Key=Project,Value=goldfish-models}]" \
    --query "Instances[0].InstanceId" --output text)
  echo "    Instance: ${INSTANCE_ID}"
fi

say "Waiting for instance to be in 'running' state…"
aws ec2 wait instance-running --instance-ids "${INSTANCE_ID}"

IP=$(aws ec2 describe-instances \
  --instance-ids "${INSTANCE_ID}" \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text)

echo
echo "================================================================="
say "Instance is up."
echo "    Instance ID:  ${INSTANCE_ID}"
echo "    Public IP:    ${IP}"
echo "    Private key:  ${KEY_PATH}"
echo
say "Next steps:"
cat <<NEXT
    # SSH in (the host key prompt will appear on first connect — type 'yes'):
    ssh -i ${KEY_PATH} ec2-user@${IP}

    # On the instance, paste the bootstrap from deploy/aws/overnight_bootstrap.sh.
    # That script:
    #   - installs dnf packages + docker + python 3.12
    #   - clones goldfish-models
    #   - pip installs deps
    #   - pulls SSM secrets into ~/.goldfish-env
    #   - smoke-tests (pytest)
    #   - launches the sweep in tmux
    #
    # Easiest way:
    scp -i ${KEY_PATH} deploy/aws/overnight_bootstrap.sh ec2-user@${IP}:~
    ssh -i ${KEY_PATH} ec2-user@${IP} 'bash ~/overnight_bootstrap.sh'

    # Later, to attach to the sweep tmux session:
    ssh -i ${KEY_PATH} ec2-user@${IP} -t tmux attach -t sweep

    # When done, terminate to stop the bill:
    aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} --region ${REGION}
NEXT
echo "================================================================="
