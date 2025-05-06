# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create log directory
RUN mkdir -p /app/logs

# Copy application code
COPY ./app ./app
COPY entrypoint.sh .

# Make entrypoint script executable
RUN chmod +x entrypoint.sh

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports for gRPC and FastAPI
EXPOSE 50051
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["./entrypoint.sh"]