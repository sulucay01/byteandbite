FROM python:3.11-slim

# Avoid .pyc and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Workdir inside container
WORKDIR /app

# System deps if you need them (edit as needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Default Cloud Run port
ENV PORT=8080

# ===== CHOOSE ONE ENTRYPOINT =====
# 1) If you have a FastAPI/Flask API:
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]