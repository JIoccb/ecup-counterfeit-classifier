
FROM python:3.10-slim

# System deps (git needed for CLIP install; optional: libgl for PIL/openCV if used)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command prints help
CMD ["python", "-m", "src.main", "--help"]
