#!/usr/bin/env bash
# Goldfish-models EC2 bootstrap.
# This script runs once on first boot via cloud-init. It:
#   1. Installs system packages (git, docker, python).
#   2. Clones the goldfish-models repo at the configured ref.
#   3. Installs Python deps + the swebench package.
#   4. Pulls API keys + HF token from SSM Parameter Store into env files.
#   5. Writes run_sweep.sh to disk.
#   6. Launches run_sweep.sh in a detached tmux session so the run
#      survives if user_data.sh exits before it finishes.
#
# All output goes to /var/log/goldfish-sweep.log. Tail it via:
#   aws ssm start-session --target <instance-id>  -> then `tail -f /var/log/goldfish-sweep.log`

set -euo pipefail

LOG=/var/log/goldfish-sweep.log
exec >>"$LOG" 2>&1
echo "===== goldfish-sweep bootstrap starting at $(date -u +%FT%TZ) ====="

# ----- System packages ---------------------------------------------------
dnf install -y git docker python3.12 python3.12-pip jq tmux unzip
systemctl enable --now docker
usermod -aG docker ec2-user

# AWS CLI v2 (AL2023 ships v2 already, but verify)
if ! command -v aws >/dev/null; then
  curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
  unzip -q /tmp/awscliv2.zip -d /tmp
  /tmp/aws/install
fi

# ----- Clone the repo ----------------------------------------------------
cd /home/ec2-user
sudo -u ec2-user git clone "@@REPO_URL@@" goldfish-models
cd goldfish-models
sudo -u ec2-user git fetch origin "@@REPO_REF@@" || true
sudo -u ec2-user git checkout "@@REPO_REF@@"

# ----- Python deps -------------------------------------------------------
python3.12 -m pip install --upgrade pip
python3.12 -m pip install -e '.[dev,api]'
python3.12 -m pip install boto3 swebench

# ----- Pull SSM secrets into env files -----------------------------------
ssm_env=/home/ec2-user/.goldfish-env
{
  for name in openai_api_key anthropic_api_key google_api_key together_api_key hf_token; do
    value=$(aws ssm get-parameter \
      --name "/goldfish/${name}" \
      --with-decryption \
      --region "@@REGION@@" \
      --query Parameter.Value \
      --output text 2>/dev/null || true)
    case "${name}" in
      openai_api_key)    var="OPENAI_API_KEY" ;;
      anthropic_api_key) var="ANTHROPIC_API_KEY" ;;
      google_api_key)    var="GOOGLE_API_KEY" ;;
      together_api_key)  var="TOGETHER_API_KEY" ;;
      hf_token)          var="HF_TOKEN" ;;
    esac
    if [[ -n "${value}" && "${value}" != "None" ]]; then
      echo "export ${var}='${value}'"
    fi
  done
  echo "export GOLDFISH_S3_BUCKET='@@BUCKET_NAME@@'"
  echo "export GOLDFISH_S3_PREFIX='runs/'"
  echo "export AWS_DEFAULT_REGION='@@REGION@@'"
} > "${ssm_env}"
chown ec2-user:ec2-user "${ssm_env}"
chmod 600 "${ssm_env}"

# ----- Write run_sweep.sh to disk ---------------------------------------
# The substitution marker below is replaced by the CDK stack at synth
# time with the full body of bootstrap/run_sweep.sh, wrapped in a
# `cat > ... <<'EOF'` heredoc. It MUST appear before the tmux launch
# below, otherwise tmux tries to source a script that doesn't exist.
@@RUN_SWEEP_EMBED@@

chown ec2-user:ec2-user /home/ec2-user/run_sweep.sh
chmod +x /home/ec2-user/run_sweep.sh

echo "===== bootstrap complete; handing off to run_sweep.sh ====="

# ----- Hand off to the sweep runner in a detached tmux session ----------
sudo -u ec2-user tmux new-session -d -s sweep \
  "bash -lc 'source ~/.goldfish-env && cd ~/goldfish-models && bash ~/run_sweep.sh; tmux wait -S sweep-done'"

# The user_data.sh process can exit; the sweep keeps running in tmux.
exit 0
