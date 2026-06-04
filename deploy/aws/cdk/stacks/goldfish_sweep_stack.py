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

import re
from pathlib import Path

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


# `cdk.Fn.sub` treats `${Name}` as a substitution variable. Anything else
# of that shape in the template would also be interpreted. Our user-data
# shell scripts use `${SHELL_VAR}` heavily, so we have to escape every
# `${...}` *except* the ones we want CFN to substitute. CFN's literal
# escape for `$` is `${!Var}`, so `${FOO}` -> `${!FOO}` keeps the literal
# shell expansion intact while CFN keeps `${BucketName}` available.
_DOLLAR_BRACE_RE = re.compile(r"\$\{([^!}][^}]*)\}")


def _protect_shell_dollar_expansions(
    template: str, *, allow: tuple[str, ...]
) -> str:
    """Escape `${VAR}` to `${!VAR}` except where VAR is in `allow`."""
    allow_set = set(allow)

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name in allow_set:
            return m.group(0)
        return "${!" + name + "}"

    return _DOLLAR_BRACE_RE.sub(repl, template)


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
        # IMPORTANT: We do template substitution in TWO stages because
        # `bucket.bucket_name` is a CFN token (a deploy-time-resolved value),
        # not a literal string at synth time. Plain str.replace breaks tokens
        # by splitting them across the surrounding string. We use
        # `cdk.Fn.sub` to do the bucket-name substitution at CFN-resolve
        # time. All other placeholders are real strings at synth time and
        # we substitute them with `str.replace` first.
        user_data_template = (BOOTSTRAP_DIR / "user_data.sh").read_text()
        run_sweep_template = (BOOTSTRAP_DIR / "run_sweep.sh").read_text()

        literal_substitutions = {
            "@@REGION@@": self.region,
            "@@REPO_URL@@": repo_url,
            "@@REPO_REF@@": repo_ref,
            "@@STACK_NAME@@": construct_id,
            "@@SWEEP_ARGS@@": sweep_args,
        }
        for placeholder, value in literal_substitutions.items():
            user_data_template = user_data_template.replace(placeholder, value)
            run_sweep_template = run_sweep_template.replace(placeholder, value)

        # Embed run_sweep.sh INSIDE user_data.sh via a heredoc. The
        # heredoc body must NOT be indented or shell parses the indent
        # as part of the file contents. CRITICALLY, the embedding must
        # happen at the @@RUN_SWEEP_EMBED@@ placeholder — which appears
        # in user_data.sh BEFORE the `tmux new-session` line, so the
        # file exists when tmux tries to source it. Appending after
        # the user_data body would put the heredoc after the `exit 0`
        # at the end of user_data.sh, which never runs.
        if "@@RUN_SWEEP_EMBED@@" not in user_data_template:
            raise RuntimeError(
                "user_data.sh template is missing the @@RUN_SWEEP_EMBED@@ "
                "placeholder; tmux launch will reference a missing file."
            )
        embedded = (
            "cat > /home/ec2-user/run_sweep.sh <<'GOLDFISH_RUN_SWEEP_EOF'\n"
            f"{run_sweep_template}"
            f"{'' if run_sweep_template.endswith(chr(10)) else chr(10)}"
            "GOLDFISH_RUN_SWEEP_EOF"
        )
        full_template = user_data_template.replace(
            "@@RUN_SWEEP_EMBED@@", embedded
        )

        # Now substitute @@BUCKET_NAME@@ via Fn::Sub so CFN resolves the
        # bucket-name token at deploy time. cdk.Fn.sub takes a string with
        # `${VarName}` placeholders + a dict of variable bindings.
        substituted_template = full_template.replace(
            "@@BUCKET_NAME@@", "${BucketName}"
        )
        # Escape any literal `${...}` already in the template so CFN
        # doesn't try to interpret them. Easiest correct way: encode `$`
        # outside our placeholder. The user_data shell scripts use a LOT
        # of `${...}` shell expansions, so this matters — we have to
        # protect each one as `${!...}` (CFN's literal-dollar escape).
        protected = _protect_shell_dollar_expansions(
            substituted_template, allow=("BucketName",)
        )
        user_data_value = cdk.Fn.sub(
            protected,
            {"BucketName": bucket.bucket_name},
        )

        user_data = ec2.UserData.custom(user_data_value)

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
