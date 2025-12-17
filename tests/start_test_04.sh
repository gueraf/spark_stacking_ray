#!/bin/bash
set -e

ADDRESS="http://10.10.20.1:8265"

echo "Submitting test_04_nvshmem.py to Ray Cluster (using uv)..."

# Ensure we are in the project root (where pyproject.toml is)
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

docker run --rm \
    -v "$(pwd):/work" \
    -w /work \
    --net=host \
    --ipc=host \
    --shm-size=10.24gb \
    rayproject/ray:latest \
    ray job submit \
    --address "$ADDRESS" \
    --runtime-env-json '{"working_dir": ".", "py_executable": "uv run python", "env_vars": {"UV_PROJECT_ENVIRONMENT": "/home/ray/anaconda3"}}' \
    -- \
    uv run python tests/test_04_nvshmem.py
