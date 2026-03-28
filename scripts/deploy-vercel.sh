#!/usr/bin/env bash
# One-shot: deploy static UI to Vercel (requires Node + network; first run opens browser login).
# Usage:
#   bash scripts/deploy-vercel.sh              # preview URL
#   bash scripts/deploy-vercel.sh --prod       # production alias
# Optional: TRACTOR_API_BASE=https://your-api.example.com

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

node scripts/inject-api-base.mjs

echo "Deploying with npx vercel (install/login on first use)..." >&2
# --archive=tgz avoids "files should NOT have more than 15000 items" when .venv exists locally
if [[ "${1:-}" == "--prod" ]]; then
  exec npx --yes vercel@latest deploy "$ROOT" --yes --archive=tgz --prod
else
  exec npx --yes vercel@latest deploy "$ROOT" --yes --archive=tgz
fi
