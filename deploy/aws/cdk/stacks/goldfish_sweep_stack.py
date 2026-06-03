"""The single stack that provisions everything for one SWE-bench sweep run.

Resources created:

  - S3 bucket for outputs (versioned, lifecycle deletes after 90 days)
  - IAM role with:
      * SSM:GetParameter on /goldfish/* (API keys + HF token live here)
      * S3 read/write on the output bucket
      * EC2 self-termination permission
  - Security group: outbound 443 + 80 (for pip / git / API calls); no
    inbound except SSH from the configured CIDR (default 0.0.0.0/32,
    i.e. none — use SSM Session Manager instead)
  - EC2 instance: c7i.4xlarge by default, 250 GB gp3 root, in the
    default VPC's default subnet
  - User-data script: pasted from `bootstrap/user_data.sh`

Required pre-flight (the operator's responsibility before `cdk deploy`):

  1. SSM parameters populated:
       /goldfish/openai_api_key      SecureString
       /goldfish/anthropic_api_key   SecureString (optional)
       /goldfish/together_api_key    SecureString (optional)
       /goldfish/hf_token            SecureString  (required for SWE-bench)
  2. AWS account has a default VPC in the chosen region.
  3. CDK is bootstrapped in the chosen account+region (`cdk bootstrap`).
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import aws_cdk as cdk
from aws_cdk import (
    Duration,
    RemovalPolicy,
    Stack,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct


HERE = Path(__file__).resolve().parent
BOOTSTRAP_DIR = HERE.parent / "bootstrap"


class GoldfishSweepStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ctx = self.node.try_get_context

        instance_type = ctx("instance_type") or "c7i.4xlarge"
        volume_size_gb = int(ctx("volume_size_gb") or 250)
        repo_url = ctx("repo_url") or "https://github.com/RSamavedam/goldfish-models.git"
        repo_ref = ctx("repo_ref") or "main"
        ami_id = ctx("ami_id")  # optional override; otherwise use AL2023
        sweep_args = ctx("sweep_args") or (
            "--config configs/sweep/phase1.yaml "
            "--output runs/phase1_shell.jsonl "
            "--limit-tasks 50 "
            "--use-swe-bench-cell yes "
            "--repo-cache-dir /home/ec2-user/repo_cache "
            "--s3-sync-every 10"
        )

        # ---------- S3 bucket for outputs ----------------------------------
        bucket = s3.Bucket(
            self,
            "OutputBucket",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-old-results",
                    enabled=True,
                    expiration=Duration.days(90),
                    noncurrent_version_expiration=Duration.days(30),
                )
            ],
        )

        # ---------- IAM role for the EC2 instance --------------------------
        role = iam.Role(
            self,
            "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="goldfish-sweep EC2 instance role",
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSSMManagedInstanceCore"
                ),
            ],
        )
        bucket.grant_read_write(role)

        # SSM: only allow GetParameter under /goldfish/*
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["ssm:GetParameter", "ssm:GetParameters"],
                resources=[
                    f"arn:aws:ssm:{self.region}:{self.account}:parameter/goldfish/*"
                ],
            )
        )
        # KMS to decrypt SecureString parameters via the AWS-managed alias.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["kms:Decrypt"],
                resources=[
                    f"arn:aws:kms:{self.region}:{self.account}:key/aws/ssm"
                ],
                conditions={
                    "StringEquals": {
                        "kms:ViaService": f"ssm.{self.region}.amazonaws.com"
                    }
                },
            )
        )

        # EC2 self-terminate permission. We scope it to instances tagged
        # with the stack name so we can't terminate unrelated instances.
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["ec2:TerminateInstances"],
                resources=[f"arn:aws:ec2:{self.region}:{self.account}:instance/*"],
                conditions={
                    "StringEquals": {
                        f"aws:ResourceTag/StackName": construct_id
                    }
                },
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["ec2:DescribeInstances", "ec2:DescribeTags"],
                resources=["*"],
            )
        )

        # ---------- Networking (default VPC) -------------------------------
        vpc = ec2.Vpc.from_lookup(self, "DefaultVpc", is_default=True)

        sg = ec2.SecurityGroup(
            self,
            "InstanceSecurityGroup",
            vpc=vpc,
            description="goldfish-sweep instance SG",
            allow_all_outbound=True,
        )
        # No SSH ingress; use SSM Session Manager:
        #   aws ssm start-session --target i-xxx

        # ---------- AMI ----------------------------------------------------
        if ami_id:
            machine_image = ec2.GenericLinuxImage(
                {self.region: ami_id}
            )
        else:
            machine_image = ec2.MachineImage.latest_amazon_linux2023()

        # ---------- User data ---------------------------------------------
        user_data_template = (BOOTSTRAP_DIR / "user_data.sh").read_text()
        run_sweep_template = (BOOTSTRAP_DIR / "run_sweep.sh").read_text()

        # Substitute placeholders so the on-instance script has the right
        # bucket name, repo URL, etc.
        substitutions = {
            "@@BUCKET_NAME@@": bucket.bucket_name,
            "@@REGION@@": self.region,
            "@@REPO_URL@@": repo_url,
            "@@REPO_REF@@": repo_ref,
            "@@STACK_NAME@@": construct_id,
            "@@SWEEP_ARGS@@": sweep_args,
        }
        for placeholder, value in substitutions.items():
            user_data_template = user_data_template.replace(placeholder, value)
            run_sweep_template = run_sweep_template.replace(placeholder, value)

        # Embed run_sweep.sh INSIDE user_data.sh via a heredoc so the
        # instance has a single bootstrap to execute.
        full_user_data = user_data_template + "\n\n" + dedent(f"""
            # ----- written by CDK: run_sweep.sh ---------------------------
            cat > /home/ec2-user/run_sweep.sh <<'GOLDFISH_RUN_SWEEP_EOF'
            {run_sweep_template}
            GOLDFISH_RUN_SWEEP_EOF
            chmod +x /home/ec2-user/run_sweep.sh
            chown ec2-user:ec2-user /home/ec2-user/run_sweep.sh
        """).strip()

        user_data = ec2.UserData.custom(full_user_data)

        # ---------- The instance -------------------------------------------
        instance = ec2.Instance(
            self,
            "SweepInstance",
            instance_type=ec2.InstanceType(instance_type),
            machine_image=machine_image,
            vpc=vpc,
            security_group=sg,
            role=role,
            user_data=user_data,
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        volume_size=volume_size_gb,
                        volume_type=ec2.EbsDeviceVolumeType.GP3,
                        delete_on_termination=True,
                        encrypted=True,
                    ),
                )
            ],
        )
        cdk.Tags.of(instance).add("StackName", construct_id)
        cdk.Tags.of(instance).add("Project", "goldfish-models")

        # ---------- Outputs ------------------------------------------------
        cdk.CfnOutput(self, "BucketName", value=bucket.bucket_name)
        cdk.CfnOutput(self, "InstanceId", value=instance.instance_id)
        cdk.CfnOutput(
            self,
            "SessionManagerCommand",
            value=f"aws ssm start-session --target {instance.instance_id}",
        )
        cdk.CfnOutput(
            self,
            "TailLogsCommand",
            value=(
                f"aws ssm start-session --target {instance.instance_id} "
                "--document-name AWS-StartInteractiveCommand "
                '--parameters command="tail -f /var/log/goldfish-sweep.log"'
            ),
        )
