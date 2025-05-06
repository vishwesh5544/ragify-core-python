#!/bin/bash
set -e

# Start FastAPI server in background
python -m app.main &

# Start gRPC server
python -m app.grpc_server