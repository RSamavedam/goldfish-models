"""CDK entry point for the goldfish-models SWE-bench sweep.

One stack:
  - S3 bucket for outputs (versioned, lifecycle-expired after 90 days)
  - IAM role for the EC2 instance with read access to SSM parameters
    (for API keys) + read/write on the output bucket
  - EC2 instance (c7i.4xlarge by default) with a 250 GB gp3 root volume
  - User-data script that bootstraps the environment, runs the sweep,
    syncs to S3, and self-terminates

Usage:
    cd deploy/aws/cdk
    pip install -r requirements.txt
    cdk synth                  # render CloudFormation; no AWS calls
    cdk deploy                 # actually create everything

The stack performs an `ec2.Vpc.from_lookup(is_default=True)` at synth
time, which requires concrete account + region values. We resolve them in
this order:
    1. `-c account=… -c region=…` CDK context overrides
    2. `CDK_DEFAULT_ACCOUNT` / `CDK_DEFAULT_REGION` env vars (the CDK CLI
       populates these from your AWS credentials automatically)
    3. `AWS_REGION` env var (region only)
    4. Hard fail with a clear message if account is missing

Other context parameters:
    instance_type     default c7i.4xlarge
    volume_size_gb    default 250
    repo_url          default https://github.com/RSamavedam/goldfish-models.git
    repo_ref          default main
    bucket_name       default (auto-generated)
    ami_id            default (latest Amazon Linux 2023 x86_64)
    sweep_args        extra args passed to scripts/sweep_shell.py
"""

from __future__ import annotations

import os
import sys

import aws_cdk as cdk

from stacks.goldfish_sweep_stack import GoldfishSweepStack


def _resolve_env(app: cdk.App) -> cdk.Environment:
    """Resolve the deploy account + region or fail loudly."""
    account = (
        app.node.try_get_context("account")
        or os.environ.get("CDK_DEFAULT_ACCOUNT")
    )
    region = (
        app.node.try_get_context("region")
        or os.environ.get("CDK_DEFAULT_REGION")
        or os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
    )
    if not account:
        print(
            "ERROR: AWS account not resolved.\n"
            "  Pass it with `-c account=379262059078`, or set\n"
            "  CDK_DEFAULT_ACCOUNT in your environment (the CDK CLI sets\n"
            "  it for you automatically if your AWS creds are configured).",
            file=sys.stderr,
        )
        sys.exit(2)
    if not region:
        print(
            "ERROR: AWS region not resolved.\n"
            "  Pass it with `-c region=us-east-2`, or set AWS_REGION /\n"
            "  CDK_DEFAULT_REGION in your environment.",
            file=sys.stderr,
        )
        sys.exit(2)
    return cdk.Environment(account=account, region=region)


def main() -> None:
    app = cdk.App()
    env = _resolve_env(app)
    GoldfishSweepStack(app, "goldfish-sweep", env=env)
    app.synth()


if __name__ == "__main__":
    main()
