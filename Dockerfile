FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TORCH_DISABLE_CUDA_FALLBACK=1

CMD ["python", "-m", "src.main", "--help"]