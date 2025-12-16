#!/bin/bash
# Start test script using 'uv run' via Ray Job submission

ADDRESS="http://127.0.0.1:8265"

echo "Submitting job to Ray Cluster at $ADDRESS..."

# We run 'ray job submit' via Docker to access the cluster network/tools
# We mount the current directory so 'test.py' and 'pyproject.toml' are available
docker run --rm \
    -v "$(pwd):/work" \
    -w /work \
    --net=host \
    rayproject/ray:latest \
    ray job submit \
    --address "$ADDRESS" \
    --runtime-env-json '{"working_dir": ".", "py_executable": "uv run"}' \
    -- \
    uv run test.py
