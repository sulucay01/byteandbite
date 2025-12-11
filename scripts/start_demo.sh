#!/bin/bash
# Start RAG Demo - Byte & Bite

echo "=========================================="
echo "Starting Byte & Bite RAG Demo"
echo "=========================================="

echo "Checking GPU availability..."
if python3 - <<'PY' 2>/dev/null
import torch
print('CUDA available:', torch.cuda.is_available())
print('CUDA device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
PY
then :; else echo "⚠ Could not check GPU status (torch may not be installed)"; fi
echo ""

if ! curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
  echo "WARNING: Ollama doesn't seem to be running!"
  echo "Please start Ollama first:  ollama serve"
  echo ""
fi

if ! curl -s http://127.0.0.1:6333/collections > /dev/null 2>&1; then
  echo "WARNING: Qdrant doesn't seem to be running!"
  echo "Please start Qdrant first"
  echo ""
fi

echo "Starting RAG backend on port 8000..."
uvicorn app:app --host 0.0.0.0 --port 8000 > /tmp/rag_backend.log 2>&1 &
BACKEND_PID=$!

echo "Waiting for backend to be ready..."
for i in {1..15}; do
  if curl -s http://127.0.0.1:8000/healthz > /dev/null 2>&1; then break; fi
  sleep 1
done

if ! curl -s http://127.0.0.1:8000/healthz > /dev/null 2>&1; then
  echo "ERROR: RAG backend failed to start!"
  echo "Check /tmp/rag_backend.log for details:"
  tail -20 /tmp/rag_backend.log
  kill $BACKEND_PID 2>/dev/null
  exit 1
fi

echo "Device status:"
curl -s http://127.0.0.1:8000/device | python3 -m json.tool 2>/dev/null || curl -s http://127.0.0.1:8000/device
echo ""
echo "Backend startup info (last lines):"
grep -E "\\[Embed\\]|device:" /tmp/rag_backend.log 2>/dev/null | tail -5 || echo "  (checking logs...)"
echo ""

echo "Starting web UI on port 5000..."
echo "Open http://localhost:5000"
echo "Press Ctrl+C to stop both servers"
echo "=========================================="
echo ""

python web_server.py

echo ""
echo "Stopping RAG backend..."
kill $BACKEND_PID 2>/dev/null
echo "Demo stopped."
