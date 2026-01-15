FROM python:3.11-slim

# Không tạo .pyc, log ra stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Cài system deps tối thiểu
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Cài uv
RUN pip install --no-cache-dir uv

# Copy file dependency trước để tận dụng cache
COPY pyproject.toml uv.lock ./

# Cài dependencies
RUN uv sync --frozen --no-dev

# Download HuggingFace model
RUN python - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
    cache_dir="/models/huggingface",
    local_files_only=False
)
EOF


# Copy source code
COPY src ./src
COPY data ./data

# Expose port
EXPOSE 8000

# Chạy FastAPI
CMD ["uvicorn", "src.app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
