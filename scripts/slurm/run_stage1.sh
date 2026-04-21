#!/usr/bin/env bash
#SBATCH --job-name=nemos-stage1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/stage1-%j.out
#SBATCH --error=logs/stage1-%j.err
# Stage 1 only calls remote NVIDIA NIM endpoints, so a CPU node is enough.
# Override per-cluster defaults with sbatch CLI flags, e.g.:
#   sbatch -p <partition> -t 01:00:00 scripts/slurm/run_stage1.sh [INPUT] [OUTPUT]
set -euo pipefail

INPUT="${1:-data/raw/sample_input.jsonl}"
OUTPUT="${2:-data/stage1/out.jsonl}"

# Under sbatch the script is copied to a spool path, so BASH_SOURCE no longer
# points at the repo. Prefer SLURM_SUBMIT_DIR (set by sbatch to the directory
# you ran sbatch from); fall back to BASH_SOURCE for direct `bash` execution.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs "$(dirname "$OUTPUT")"

# sbatch runs a non-login shell, so ~/.local/bin (where uv usually lives) is
# not in PATH. Add it and fail fast if uv is still missing.
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || { echo "uv not found; install it or adjust PATH" >&2; exit 127; }

# .env is the only source for NVIDIA_API_KEY (stage1 loads it via python-dotenv).
[[ -f .env ]] || { echo ".env missing in $REPO_ROOT — copy .env.example and set NVIDIA_API_KEY" >&2; exit 2; }

echo "[stage1] host=$(hostname) input=$INPUT output=$OUTPUT"
uv run python scripts/run_stage.py --stage 1 --input "$INPUT" --output "$OUTPUT"
