#!/usr/bin/env bash
# botkeep entrypoint
set -euo pipefail
echo "[boot] installing python deps..."
pip install --no-cache-dir --upgrade pip >/dev/null
pip install --no-cache-dir -r requirements.txt

echo "[boot] starting server on port ${PORT:-8080}..."
exec python -m uvicorn app:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers 1 \
  --timeout-keep-alive 30
