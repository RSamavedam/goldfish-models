#!/usr/bin/env bash
# Goldfish-models on-instance bootstrap.
# Runs ONCE on the EC2 instance after `launch_overnight.sh` provisioned it.
#
# Copy this to the instance:
#   scp -i ~/goldfish-overnight.pem deploy/aws/overnight_bootstrap.sh ec2-user@<IP>:~
#
# Then run it from your laptop OR after sshing in:
#   bash overnight_bootstrap.sh
#
# What it does:
#   1. Installs dnf packages + docker + python 3.12.
#   2. Clones the goldfish-models repo.
#   3. pip installs Python deps.
#   4. Pulls secrets from SSM into ~/.goldfish-env.
#   5. Runs pytest smoke (fails fast if anything is wrong).
#   6. Launches the sweep in a detached tmux session named `sweep`.

set -euo pipefail

REGION="${REGION:-us-east-2}"
REPO_URL="${REPO_URL:-https://github.com/RSamavedam/goldfish-models.git}"
REPO_REF="${REPO_REF:-main}"
SWEEP_CONFIG="${SWEEP_CONFIG:-configs/sweep/swe_bench.yaml}"
LIMIT_TASKS="${LIMIT_TASKS:-50}"

say() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

# --------------------------------------------------------------------- #
# Packages                                                              #
# --------------------------------------------------------------------- #
say "Installing system packages (dnf)…"
# patch: AL2023 doesn't ship it by default; SWE-bench models reach for
#        it to apply diffs. (Bug 11.)
# nl, less, more: handy text utilities models try; harmless to include.
sudo dnf install -y git docker python3.12 python3.12-pip jq tmux unzip \
                     patch less coreutils
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user

# --------------------------------------------------------------------- #
# Repo                                                                  #
# --------------------------------------------------------------------- #
cd ~
if [[ ! -d goldfish-models ]]; then
  say "Cloning ${REPO_URL}…"
  git clone "${REPO_URL}" goldfish-models
fi
cd goldfish-models
git fetch origin "${REPO_REF}" || true
git checkout "${REPO_REF}"
git pull --ff-only origin "${REPO_REF}" || true

# --------------------------------------------------------------------- #
# Python deps                                                           #
# --------------------------------------------------------------------- #
say "Installing Python deps (this takes ~2 min)…"
sudo python3.12 -m pip install --upgrade pip
sudo python3.12 -m pip install -e '.[dev,api]'
sudo python3.12 -m pip install boto3 swebench

# --------------------------------------------------------------------- #
# Secrets from SSM                                                      #
# --------------------------------------------------------------------- #
say "Pulling secrets from SSM Parameter Store…"
ssm_env=~/.goldfish-env
get_param() {
  aws ssm get-parameter \
    --name "$1" --with-decryption --region "${REGION}" \
    --query Parameter.Value --output text 2>/dev/null || true
}

OPENAI_API_KEY=$(get_param /goldfish/openai_api_key)
HF_TOKEN=$(get_param /goldfish/hf_token)
ANTHROPIC_API_KEY=$(get_param /goldfish/anthropic_api_key)
TOGETHER_API_KEY=$(get_param /goldfish/together_api_key)

[[ -z "${OPENAI_API_KEY}" || "${OPENAI_API_KEY}" == "None" ]] && {
  echo "ERROR: /goldfish/openai_api_key missing from SSM" >&2; exit 1; }
[[ -z "${HF_TOKEN}" || "${HF_TOKEN}" == "None" ]] && {
  echo "ERROR: /goldfish/hf_token missing from SSM" >&2; exit 1; }

cat > "${ssm_env}" <<EOF
export OPENAI_API_KEY='${OPENAI_API_KEY}'
export HF_TOKEN='${HF_TOKEN}'
EOF
if [[ -n "${ANTHROPIC_API_KEY}" && "${ANTHROPIC_API_KEY}" != "None" ]]; then
  echo "export ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY}'" >> "${ssm_env}"
fi
if [[ -n "${TOGETHER_API_KEY}" && "${TOGETHER_API_KEY}" != "None" ]]; then
  echo "export TOGETHER_API_KEY='${TOGETHER_API_KEY}'" >> "${ssm_env}"
fi
echo "export AWS_DEFAULT_REGION='${REGION}'" >> "${ssm_env}"
chmod 600 "${ssm_env}"
echo "    Wrote ${ssm_env}"

# --------------------------------------------------------------------- #
# Smoke test                                                            #
# --------------------------------------------------------------------- #
say "Running pytest smoke (no API calls)…"
cd ~/goldfish-models
if ! PYTHONPATH=src python3.12 -m pytest tests/ -q 2>&1 | tee /tmp/pytest.log | tail -5; then
  say "pytest reported failures; see /tmp/pytest.log for the full output."
  say "Aborting before we spend money on a broken harness."
  exit 1
fi

# --------------------------------------------------------------------- #
# Launch sweep                                                          #
# --------------------------------------------------------------------- #
say "Launching sweep in detached tmux session 'sweep'…"
mkdir -p ~/goldfish-models/runs

# Make sure there isn't a stale session from a previous run.
if tmux has-session -t sweep 2>/dev/null; then
  say "Existing tmux session 'sweep' found; killing it before relaunch."
  tmux kill-session -t sweep
fi

tmux new-session -d -s sweep "bash -lc '
  set -euo pipefail
  source ~/.goldfish-env
  cd ~/goldfish-models
  PYTHONPATH=src python3.12 scripts/sweep_shell.py \
    --config ${SWEEP_CONFIG} \
    --output runs/phase1_shell.jsonl \
    --limit-tasks ${LIMIT_TASKS} \
    --use-swe-bench-cell yes \
    --repo-cache-dir ~/repo_cache \
    --s3-sync-every 10 \
    2>&1 | tee ~/sweep.log
'"

echo
echo "================================================================="
say "Bootstrap complete. Sweep is running in tmux session 'sweep'."
echo
echo "    Tail the log:        tmux attach -t sweep   (detach: Ctrl-b d)"
echo "    Or from your laptop: ssh -i <key> ec2-user@<ip> -t tmux attach -t sweep"
echo "    The log is also at:  ~/sweep.log"
echo "    JSONL output:        ~/goldfish-models/runs/phase1_shell.jsonl"
echo
echo "    When done:           sudo shutdown -h now"
echo "                         (or terminate from your laptop:"
echo "                          aws ec2 terminate-instances --instance-ids <id>)"
echo "================================================================="
