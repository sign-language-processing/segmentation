FROM python:3.12-slim

ENV PYTHONUNBUFFERED=True

WORKDIR /app

# Install torch CPU-only first (smaller image, inference doesn't need CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install third-party dependencies (cached layer — only re-runs if pyproject.toml changes)
COPY pyproject.toml README.md ./
RUN mkdir -p sign_language_segmentation/model sign_language_segmentation/data && \
    touch sign_language_segmentation/__init__.py \
          sign_language_segmentation/model/__init__.py \
          sign_language_segmentation/data/__init__.py && \
    pip install --no-cache-dir ".[server]" && \
    rm -rf sign_language_segmentation

# Copy source (includes dist/2026/best.ckpt as package data) and install package
COPY sign_language_segmentation ./sign_language_segmentation
RUN pip install --no-cache-dir --no-deps -e .

# Warm up: run inference once so model is loaded and cached for the first real request
RUN pose_to_segments \
      --pose sign_language_segmentation/tests/example.pose \
      --elan /tmp/warmup.eaf \
      --no-pose-link && \
    rm /tmp/warmup.eaf

CMD ["sh", "-c", "exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 sign_language_segmentation.server:app"]
