#!/bin/bash
# Start test script using 'uv run' via Ray Job submission

INFO_FILE="ray_info.txt"

if [ ! -f "$INFO_FILE" ]; then
    echo "Error: $INFO_FILE not found. Is the cluster running?"
    exit 1
fi

HEAD_IP=$(cat "$INFO_FILE")
ADDRESS="http://$HEAD_IP:8265"

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