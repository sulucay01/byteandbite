#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "Starting Byte & Bite"
echo "=========================================="

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
UI_PORT="${PORT:-8080}"              # Cloud Run 8080 ister, VM'de de kullanılır
RAG_BACKEND_URL="${RAG_BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT}}"

export RAG_BACKEND_URL

echo "[Info] RAG_BACKEND_URL=${RAG_BACKEND_URL}"
echo "[Info] UI_PORT=${UI_PORT}"
echo ""

# Optional GPU check (won't fail if torch not installed)
python3 - <<'PY' 2>/dev/null || true
import torch
print("CUDA:", torch.cuda.is_available())
PY

echo ""
echo "[1/2] Starting Backend (FastAPI)..."
# Uvicorn'u background başlat
uvicorn app:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" --log-level info &
BACKEND_PID=$!

# Çıkışta temiz kapat
cleanup () {
  echo ""
  echo "[Cleanup] Stopping backend..."
  kill "${BACKEND_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[Wait] Backend health check..."
for i in $(seq 1 30); do
  if curl -fsS "${RAG_BACKEND_URL}/healthz" >/dev/null 2>&1; then
    echo "[OK] Backend UP!"
    break
  fi
  sleep 1
  if [ "$i" -eq 30 ]; then
    echo "[ERR] Backend did not become ready."
    exit 1
  fi
done

echo ""
echo "[2/2] Starting Web UI (Flask)..."
# Flask web_server.py zaten PORT env varını okuyacak şekilde ayarlı olmalı
python3 web_server.py
