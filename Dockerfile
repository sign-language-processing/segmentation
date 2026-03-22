FROM python:3.12-slim

ENV PYTHONUNBUFFERED True

WORKDIR /app

# Install torch CPU-only first (smaller image, inference doesn't need CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
COPY pyproject.toml .
RUN mkdir -p sign_language_segmentation/model sign_language_segmentation/data && \
    touch sign_language_segmentation/__init__.py \
          sign_language_segmentation/model/__init__.py \
          sign_language_segmentation/data/__init__.py \
          README.md && \
    pip install --no-cache-dir ".[server]"

# Copy source and model checkpoint
COPY sign_language_segmentation ./sign_language_segmentation
COPY dist ./dist

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 sign_language_segmentation.server:app
