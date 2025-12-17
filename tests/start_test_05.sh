#!/bin/bash
set -e

ADDRESS="http://10.10.20.1:8265"

echo "Submitting test_05_symm_mem_nccl.py to Ray Cluster (using uv)..."

# Ensure we are in the project root (where pyproject.toml is)
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Note: We probably don't STRICTLY need --ipc=host for NCCL backend if it doesn't use the FD passing mechanism of NVSHMEM? 
# But it might still use SHM. Let's start WITHOUT special flags to see if NCCL mode is more robust out-of-the-box, 
# as that would be a key finding for the user.
# UPDATE: Actually, standard NCCL usually needs SHM. 
# But let's try standard container first (test_03 used standard and worked for basic NCCL).

docker run --rm \
    -v "$(pwd):/work" \
    -w /work \
    --net=host \
    rayproject/ray:latest \
    ray job submit \
    --address "$ADDRESS" \
    --runtime-env-json '{"working_dir": ".", "py_executable": "uv run python", "env_vars": {"UV_PROJECT_ENVIRONMENT": "/home/ray/anaconda3"}}' \
    -- \
    uv run python tests/test_05_symm_mem_nccl.py
