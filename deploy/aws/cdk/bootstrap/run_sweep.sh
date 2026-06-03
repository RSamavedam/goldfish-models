#!/usr/bin/env bash
# Main sweep runner. Executed inside a detached tmux session by user_data.sh.
# Streams everything to /var/log/goldfish-sweep.log and to tmux scrollback.

set -euo pipefail
LOG=/var/log/goldfish-sweep.log
exec >>"$LOG" 2>&1
echo "===== run_sweep.sh starting at $(date -u +%FT%TZ) ====="

source ~/.goldfish-env
cd ~/goldfish-models

mkdir -p runs

# ----- Tests --------------------------------------------------------------
# Smoke test before we burn API money. Skip Docker-bound tests; they need
# real images.
echo "----- pytest smoke -----"
PYTHONPATH=src python3.12 -m pytest tests/ -q --ignore=tests/swe_bench_docker 2>&1 | tail -10 || {
  echo "WARNING: pytest reported failures; continuing anyway"
}

# ----- Final-sweep upload helper ------------------------------------------
upload_results() {
  if [[ -n "${GOLDFISH_S3_BUCKET:-}" ]]; then
    aws s3 sync runs/ "s3://${GOLDFISH_S3_BUCKET}/${GOLDFISH_S3_PREFIX:-runs/}" \
      --no-progress || echo "(s3 sync failed)"
  fi
}

# Make sure results upload even if the sweep crashes.
trap 'upload_results' EXIT

# ----- The sweep ----------------------------------------------------------
echo "----- launching sweep -----"
PYTHONPATH=src python3.12 scripts/sweep_shell.py @@SWEEP_ARGS@@ || {
  echo "WARNING: sweep exited non-zero"
}

echo "===== sweep done at $(date -u +%FT%TZ) ====="

# ----- Self-terminate -----------------------------------------------------
# The IAM policy only allows TerminateInstances against instances tagged
# StackName=@@STACK_NAME@@, so this is a no-op (with logged error) if the
# tags or policy aren't right.
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/instance-id)
echo "Self-terminating instance ${INSTANCE_ID}"
aws ec2 terminate-instances \
  --instance-ids "${INSTANCE_ID}" \
  --region "@@REGION@@" || echo "self-terminate failed; will need manual cleanup"
