#!/bin/bash
set -e

# Adapted from https://github.com/eugr/spark-vllm-docker/blob/main/run-cluster-node.sh

export_persist() {
    local var_name="$1"
    local var_value="$2"
    export "$var_name"="$var_value"
    # Append to .bashrc for future debugging
    if ! grep -q "export $var_name=" ~/.bashrc; then
        echo "export $var_name=\"$var_value\"" >> ~/.bashrc
    else
        sed -i "s|export $var_name=.*|export $var_name=\"$var_value\"|" ~/.bashrc
    fi
}

usage() {
    echo "Usage: $0 --role <head|node> --host-ip <ip> --eth-if <name> --ib-if <name> [--head-ip <ip>]"
    exit 1
}

NODE_TYPE=""
HOST_IP=""
ETH_IF_NAME=""
IB_IF_NAME=""
HEAD_IP=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -r|--role) NODE_TYPE="$2"; shift ;;
        -h|--host-ip) HOST_IP="$2"; shift ;;
        -e|--eth-if) ETH_IF_NAME="$2"; shift ;;
        -i|--ib-if) IB_IF_NAME="$2"; shift ;;
        -m|--head-ip) HEAD_IP="$2"; shift ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
    shift
done

if [[ -z "$NODE_TYPE" || -z "$HOST_IP" || -z "$ETH_IF_NAME" || -z "$IB_IF_NAME" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

if [[ "$NODE_TYPE" == "node" && -z "$HEAD_IP" ]]; then
    echo "Error: --head-ip required for node role."
    exit 1
fi

echo "Configuring environment for [$NODE_TYPE] at $HOST_IP..."

# Env Vars from eugr repo
export_persist VLLM_HOST_IP "$HOST_IP"
export_persist RAY_NODE_IP_ADDRESS "$HOST_IP"
export_persist RAY_OVERRIDE_NODE_IP_ADDRESS "$HOST_IP"

# Network
export_persist MN_IF_NAME "$ETH_IF_NAME"
export_persist UCX_NET_DEVICES "$ETH_IF_NAME"
export_persist NCCL_SOCKET_IFNAME "$ETH_IF_NAME"

# InfiniBand
export_persist NCCL_IB_HCA "$IB_IF_NAME"
export_persist NCCL_IB_DISABLE "0"

# Sockets
export_persist OMPI_MCA_btl_tcp_if_include "$ETH_IF_NAME"
export_persist GLOO_SOCKET_IFNAME "$ETH_IF_NAME"
export_persist TP_SOCKET_IFNAME "$ETH_IF_NAME"
export_persist RAY_memory_monitor_refresh_ms "0"

# Add CUDA to PATH just in case
export PATH=/usr/local/cuda/bin:$PATH

echo ">>> Starting Ray..."

if [ "${NODE_TYPE}" == "head" ]; then
    # Start Head
    exec ray start --block --head --port 6379 \
        --node-ip-address "$HOST_IP" \
        --include-dashboard=True \
        --dashboard-host "0.0.0.0" \
        --dashboard-port 8265 \
        --disable-usage-stats
else
    # Start Worker
    exec ray start --block \
        --address="$HEAD_IP:6379" \
        --node-ip-address "$HOST_IP" \
        --disable-usage-stats
fi
