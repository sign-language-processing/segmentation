FROM python:3.12-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Setup local workdir and dependencies
WORKDIR /app

# Install python dependencies.
ADD ./pyproject.toml ./pyproject.toml
RUN mkdir -p sign_language_segmentation/src/utils
RUN touch README.md
RUN pip install --no-cache-dir ".[server]"

# Copy local code to the container image.
COPY ./sign_language_segmentation ./sign_language_segmentation

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 sign_language_segmentation.server:app
