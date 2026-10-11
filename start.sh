#!/usr/bin/env bash
# botkeep entrypoint.
# botkeep exposes the port via SERVER_PORT (their canonical env var).
# Fall back to PORT for dev / other PaaS (Render, Railway, etc.).
set -euo pipefail

echo "[boot] installing python deps..."
pip install --no-cache-dir --upgrade pip >/dev/null
pip install --no-cache-dir -r requirements.txt

# Resolve port: SERVER_PORT (botkeep) > PORT (generic) > 8080
SVC_PORT="${SERVER_PORT:-${PORT:-8080}}"
echo "[boot] starting server on port ${SVC_PORT}..."

exec python -m uvicorn app:app \
  --host 0.0.0.0 \
  --port "${SVC_PORT}" \
  --workers 1 \
  --timeout-keep-alive 30
