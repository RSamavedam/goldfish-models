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

Parameters (cdk context):
    instance_type     default c7i.4xlarge
    volume_size_gb    default 250
    repo_url          default https://github.com/RSamavedam/goldfish-models.git
    repo_ref          default main
    bucket_name       default (auto-generated)
    ami_id            default (latest Amazon Linux 2023 x86_64)
    sweep_args        extra args passed to scripts/sweep_shell.py
"""

from __future__ import annotations

import aws_cdk as cdk

from stacks.goldfish_sweep_stack import GoldfishSweepStack


def main() -> None:
    app = cdk.App()
    GoldfishSweepStack(
        app,
        "goldfish-sweep",
        env=cdk.Environment(
            account=app.node.try_get_context("account") or None,
            region=app.node.try_get_context("region") or "us-east-1",
        ),
    )
    app.synth()


if __name__ == "__main__":
    main()
