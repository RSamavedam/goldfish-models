# Goldfish-models · AWS deployment playbook

Provisions a single EC2 instance that runs the SWE-bench Verified sweep
end-to-end, periodically syncs results to S3, and self-terminates when
done. Total wall-clock for the conservative 3×4×50 sweep is ~24 hours
on a `c7i.4xlarge`. API spend estimate: $30–60 (mostly OpenAI o-series
reasoning tokens).

## Pre-flight (one-time, before first deploy)

### 1. AWS credentials + region

Make sure your AWS CLI is configured against the right account:

```bash
aws sts get-caller-identity        # confirms who you are
export AWS_REGION=us-east-1        # or whichever region you want
```

### 2. Bootstrap CDK (one-time per account/region)

```bash
cdk bootstrap aws://$(aws sts get-caller-identity --query Account --output text)/$AWS_REGION
```

### 3. Populate SSM Parameter Store

The instance reads API keys from SSM at boot. Populate them as
`SecureString` parameters under `/goldfish/`:

```bash
aws ssm put-parameter --name /goldfish/openai_api_key \
  --type SecureString --value "sk-..." --overwrite

aws ssm put-parameter --name /goldfish/hf_token \
  --type SecureString --value "hf_..." --overwrite

# Optional (only if you want to sweep these providers too):
aws ssm put-parameter --name /goldfish/anthropic_api_key \
  --type SecureString --value "sk-ant-..." --overwrite
aws ssm put-parameter --name /goldfish/together_api_key \
  --type SecureString --value "..." --overwrite
```

The HF token is **required** — SWE-bench Verified is a gated dataset on
HuggingFace. Sign in there and accept the dataset terms first:
https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified

### 4. Install CDK locally

```bash
cd deploy/aws/cdk
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
npm install -g aws-cdk        # if you don't already have the cdk CLI
```

## Deploy

```bash
cd deploy/aws/cdk
source .venv/bin/activate

# Inspect the synthesized template first:
cdk synth

# Apply:
cdk deploy
```

The stack outputs include the bucket name and the instance ID. The
instance starts running the sweep within ~5 minutes of `cdk deploy`
completing (first-boot package installs take a few minutes).

## Monitor while it's running

The instance doesn't accept SSH; use SSM Session Manager:

```bash
INSTANCE_ID=$(aws cloudformation describe-stacks \
  --stack-name goldfish-sweep \
  --query 'Stacks[0].Outputs[?OutputKey==`InstanceId`].OutputValue' \
  --output text)

# Tail the sweep log:
aws ssm start-session --target $INSTANCE_ID
# then on the instance:
tail -f /var/log/goldfish-sweep.log

# Attach to the live tmux session:
sudo -u ec2-user tmux attach -t sweep
# (detach with Ctrl-b d)
```

## Retrieve results

The instance uploads `runs/phase1_shell.jsonl` to S3 every 10 completed
cells, and again at the end via `aws s3 sync`. To pull results locally:

```bash
BUCKET=$(aws cloudformation describe-stacks \
  --stack-name goldfish-sweep \
  --query 'Stacks[0].Outputs[?OutputKey==`BucketName`].OutputValue' \
  --output text)

aws s3 sync "s3://$BUCKET/runs/" ./runs_from_cloud/
PYTHONPATH=src python scripts/analyze.py ./runs_from_cloud/phase1_shell.jsonl
```

## Teardown

The instance self-terminates when the sweep completes. The S3 bucket is
retained (results survive). To remove everything:

```bash
# Empty the bucket first (CDK refuses to delete a non-empty bucket).
aws s3 rm "s3://$BUCKET/" --recursive
cdk destroy
```

## Cost knobs

The stack reads context overrides for the most likely things to tune:

```bash
cdk deploy \
  -c instance_type=c7i.8xlarge \
  -c volume_size_gb=500 \
  -c repo_ref=main \
  -c sweep_args="--config configs/sweep/swe_bench.yaml --limit-tasks 30 --use-swe-bench-cell yes --s3-sync-every 5"
```

- **`instance_type`** — `c7i.4xlarge` (default) ≈ $0.71/hr. `c7i.8xlarge`
  for faster Docker-based scoring (~2× the wall-clock improvement; not
  exactly 2× because the bottleneck is API latency, not local compute).
- **`volume_size_gb`** — 250 GB is enough for ~50 SWE-bench Docker
  images. Bump to 500 GB if you sweep more repos.
- **`repo_ref`** — git ref of `goldfish-models` to deploy. `main` by
  default; use a tag or SHA to pin.
- **`sweep_args`** — passed through to `scripts/sweep_shell.py`. Use this
  to change the config, limit tasks, switch L values, etc.

## What can go wrong

A non-exhaustive list of failure modes I've thought about but haven't
hit yet (this is the first cloud deploy):

- **Model identifiers wrong.** The default config uses `gpt-5`,
  `o3-mini`, `o4-mini`. If your account doesn't expose those exact
  names, the harness logs per-cell `provider_error` and continues. The
  sweep completes but every cell fails. Catch this by tailing the log
  for the first ~3 cells.
- **HF gate not accepted.** Loader fails with "401 Client Error" on
  `princeton-nlp/SWE-bench_Verified`. Fix: accept the dataset terms on
  HF, then redeploy or re-run by re-attaching to the instance.
- **Docker image pulls slow.** First-task wall-clock includes pulling
  the per-repo Docker image (~minutes). Subsequent tasks against the
  same repo reuse the cache. Patience.
- **Self-terminate fails.** The IAM policy scopes
  `ec2:TerminateInstances` to instances tagged `StackName=goldfish-sweep`.
  If the tag isn't propagating (rare), the instance stays running.
  Manually terminate via console or:
  `aws ec2 terminate-instances --instance-ids $INSTANCE_ID`.
- **EBS full.** SWE-bench images grow over time. Bump
  `volume_size_gb` if you see "no space left on device" in the log.
- **OpenAI rate limits.** The harness retries with exponential backoff
  (see `src/rlm_paged/client/_retry.py`), but a sustained rate-limit
  storm will slow the sweep. Lower `--max_workers` (passed to the
  SWE-bench scorer) or wait it out.
