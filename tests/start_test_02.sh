#!/bin/bash
# Run from project root

ADDRESS="http://10.10.20.1:8265"

echo "Submitting test_02_gpu_check_uv.py to Ray Cluster (using uv)..."

# Ensure we are in the project root (where pyproject.toml is)
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

docker run --rm \
    -v "$(pwd):/work" \
    -w /work \
    --net=host \
    rayproject/ray:latest \
    ray job submit \
    --address "$ADDRESS" \
    --runtime-env-json '{"working_dir": ".", "py_executable": "uv run python", "env_vars": {"UV_PROJECT_ENVIRONMENT": "/home/ray/anaconda3"}}' \
    -- \
    uv run tests/test_02_gpu_check_uv.py
