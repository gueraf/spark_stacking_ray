#!/bin/bash
set -e

# Load Config
export KUBECONFIG=~/.kube/config_k3s_custom

echo ">>> Deploying NVIDIA Device Plugin (if needed)..."
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/4f1de00eb6eb148dcd90cf85e9ce0fc830b43651/deployments/static/nvidia-device-plugin.yml
kubectl apply -f nvidia_runtime_class.yaml

echo ">>> Cleaning up old jobs..."
kubectl delete pytorchjob pytorch-gpu-test 2>/dev/null || true
# Force delete pods if they are stuck
kubectl delete pods -l pytorch-job-name=pytorch-gpu-test --force --grace-period=0 2>/dev/null || true

echo ">>> Submitting GPU Test Job..."
kubectl apply -f kubeflow_gpu_test.yaml

echo ">>> Waiting for Pods to be created..."
sleep 5
kubectl get pods -l pytorch-job-name=pytorch-gpu-test -o wide

echo ">>> Tailing logs (Ctrl+C to stop)..."
# Wait a bit for containers to actually start
sleep 5
kubectl logs -l pytorch-job-name=pytorch-gpu-test -f --prefix=true
