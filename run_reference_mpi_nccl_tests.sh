#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Set environment variables
export CUDA_HOME="/usr/local/cuda"
export MPI_HOME="/usr/lib/aarch64-linux-gnu/openmpi"
export NCCL_HOME="$HOME/nccl/build/"
export LD_LIBRARY_PATH="$NCCL_HOME/lib:$CUDA_HOME/lib64/:$MPI_HOME/lib:$LD_LIBRARY_PATH"

# Set network interface environment variables
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1

echo "Building NCCL..."
# Build NCCL
cd "$HOME/nccl"
make -j

echo "Building NCCL Tests..."
# Build NCCL Tests
cd "$HOME/nccl-tests"
make -j

echo "Running all_gather performance test..."
# Run the all_gather performance test
mpirun -np 2 -H yin-zrh:1,yang-zrh:1 \
  --mca plm_rsh_agent "ssh -o StrictHostKeyChecking=no" \
  -x LD_LIBRARY_PATH=$LD_LIBRARY_PATH \
  "$HOME/nccl-tests/build/all_gather_perf"
