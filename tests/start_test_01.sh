#!/bin/bash
# Run from project root

ADDRESS="http://127.0.0.1:8265"

echo "Submitting test_01_gpu_check.py to Ray Cluster..."

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
    --runtime-env-json '{"working_dir": "."}' \
    -- \
    python tests/test_01_gpu_check.py
