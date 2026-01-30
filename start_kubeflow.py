#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import threading
import argparse
import getpass

# We reuse the discovery logic from start.py to ensure consistency
# In a real scenario, we might import this, but for a standalone script we copy the relevant parts.

def run_ssh_cmd(ip, cmd, check=True, stream_output=False, env=None):
    # Use fabian as the default user if not specified in ~/.ssh/config, 
    # but usually 'ssh ip' uses the current user. 
    # The user confirmed the account is 'fabian'.
    ssh_cmd = ["ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", ip, cmd]
    
    # Propagate environment variables if needed, though SSH usually needs explicit exporting in the command string
    # For now, we assume cmd string handles exports if needed.
    
    if stream_output:
        return subprocess.run(ssh_cmd, check=check, env=env)
    else:
        return subprocess.run(ssh_cmd, check=check, capture_output=True, text=True, env=env)

def discover_nodes(auto_confirm=False):
    print("Discovering nodes (using Avahi)...")
    
    # 1. Identify Preferred Interface (Local)
    preferred_iface = None
    try:
        out = subprocess.check_output("ibdev2netdev | grep 'Up' | grep -v 'p2p'", shell=True).decode()
        for line in out.splitlines():
            if "==>" in line:
                preferred_iface = line.split("==>")[1].strip().split()[0]
                print(f"Preferred Interface: {preferred_iface}")
                break
    except:
        pass

    try:
         # 2. Avahi Discovery
         out = subprocess.check_output("avahi-browse -p -r -f -t _ssh._tcp", shell=True).decode()
         ips = set()
         
         for line in out.splitlines():
             if not line.startswith("="): continue
             parts = line.split(";")
             if len(parts) > 7:
                 iface = parts[1]
                 ip = parts[7]
                 if ":" in ip: continue
                 if preferred_iface and iface != preferred_iface:
                     continue
                 ips.add(ip)
         
         nodes = sorted(list(ips))
    except Exception as e:
        print(f"Discovery failed: {e}")
        nodes = ["127.0.0.1"]

    print(f"Found nodes: {nodes}")
    
    if auto_confirm:
        print("Auto-confirming node list.")
        return nodes

    while True:
        c = input("Confirm node list for K3s Cluster (Y/n/manual): ").strip().lower()
        if c == '' or c == 'y': return nodes
        if c == 'n': sys.exit(0)
        if c == 'manual':
             m = []
             print("Enter IPs (Empty to finish):")
             while True:
                 x = input("> ").strip()
                 if not x: break
                 m.append(x)
             return m

def install_k3s_head(ip, sudo_pass):
    print(f"[{ip}] Installing K3s Master (Head)...")
    # 1. Download script
    # 2. Run with sudo (piping password)
    # 3. Clean up
    # We add --tls-san {ip} to ensure the certificate covers the interface we are using to connect (e.g. 10.10.20.1)
    install_cmd = (
        "curl -sfL https://get.k3s.io -o k3s_install.sh && "
        f"echo '{sudo_pass}' | sudo -S sh k3s_install.sh --docker --write-kubeconfig-mode 644 --tls-san {ip} && "
        "rm k3s_install.sh"
    )
    run_ssh_cmd(ip, install_cmd, stream_output=True)
    
    # Retrieve Token (with retry)
    print(f"[{ip}] Retrieving Cluster Token...")
    token_cmd = f"echo '{sudo_pass}' | sudo -S cat /var/lib/rancher/k3s/server/node-token"
    
    for attempt in range(10):
        try:
            res = run_ssh_cmd(ip, token_cmd, check=True)
            token = res.stdout.strip()
            if token:
                return token
        except subprocess.CalledProcessError:
            pass
        
        print(f"[{ip}] Waiting for token file (attempt {attempt+1}/10)...")
        time.sleep(2)
        
    raise Exception(f"[{ip}] Failed to retrieve K3s node token after installation.")

def install_k3s_worker(ip, head_ip, token, sudo_pass):
    print(f"[{ip}] Installing K3s Agent (Worker)...")
    
    # 0. Clean up any bad config from previous runs to ensure clean install
    cleanup_cmd = f"echo '{sudo_pass}' | sudo -S rm -f /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl"
    try:
        run_ssh_cmd(ip, cleanup_cmd, stream_output=False)
    except:
        pass # Ignore if file doesn't exist

    # Pass env vars to the shell being run by sudo
    install_cmd = (
        "curl -sfL https://get.k3s.io -o k3s_install.sh && "
        f"echo '{sudo_pass}' | sudo -S K3S_URL=https://{head_ip}:6443 K3S_TOKEN={token} sh k3s_install.sh --docker && "
        "rm k3s_install.sh"
    )
    run_ssh_cmd(ip, install_cmd, stream_output=True)

def setup_kubeflow_training_operator(head_ip, sudo_pass):
    print("\n>>> Setting up Kubeflow Training Operator...")
    # Use kubectl apply -k with --server-side to avoid "metadata.annotations: Too long" error
    # This is common with large CRDs like PyTorchJob
    manifest_target = "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.9.0"
    
    # We run kubectl on the head node directly for simplicity
    cmd = f"echo '{sudo_pass}' | sudo -S kubectl apply --server-side --force-conflicts -k \"{manifest_target}\""
    run_ssh_cmd(head_ip, cmd, stream_output=True)
    
    print("Waiting for Training Operator to be ready...")
    # Simple wait loop
    time.sleep(10)
    check_cmd = f"echo '{sudo_pass}' | sudo -S kubectl get pods -n kubeflow -l control-plane=training-operator-controller-manager"
    run_ssh_cmd(head_ip, check_cmd, stream_output=True)

def scp_file(ip, local_path, remote_path):
    cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", local_path, f"{ip}:{remote_path}"]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def build_node_image(ip):
    REMOTE_BUILD_DIR = "/tmp/ray_build"
    DOCKER_IMAGE = "ray_nccl_node"
    DOCKERFILE = "Dockerfile"
    RUN_SCRIPT = "run_node.sh"

    print(f"[{ip}] Preparing build context...")
    run_ssh_cmd(ip, f"mkdir -p {REMOTE_BUILD_DIR}")
    scp_file(ip, DOCKERFILE, f"{REMOTE_BUILD_DIR}/Dockerfile")
    scp_file(ip, RUN_SCRIPT, f"{REMOTE_BUILD_DIR}/run_node.sh")
    
    print(f"[{ip}] Building Docker image (this may take a while)...")
    res = run_ssh_cmd(ip, f"cd {REMOTE_BUILD_DIR} && docker build -t {DOCKER_IMAGE} .", stream_output=True)
    if res.returncode != 0:
        print(f"[{ip}] Build FAILED.")
    else:
        print(f"[{ip}] Build Success.")

def configure_nvidia_k3s(ip, sudo_pass):
    print(f"[{ip}] Configuring NVIDIA Runtime for Docker...")
    
    # Since we are using K3s with --docker, we need to configure the Docker daemon
    # to support the NVIDIA runtime.
    
    # 1. Configure Docker Daemon using nvidia-ctk
    # This modifies /etc/docker/daemon.json
    # We add --set-as-default so K3s/Kubernetes uses it for all pods (including Device Plugin)
    cmd_config = f"echo '{sudo_pass}' | sudo -S nvidia-ctk runtime configure --runtime=docker --set-as-default"
    run_ssh_cmd(ip, cmd_config, stream_output=True)

    # 2. Restart Docker to apply changes
    print(f"[{ip}] Restarting Docker...")
    cmd_restart_docker = f"echo '{sudo_pass}' | sudo -S systemctl restart docker"
    run_ssh_cmd(ip, cmd_restart_docker, stream_output=True)

    # 3. Restart K3s to reconnect to Docker
    print(f"[{ip}] Restarting K3s service...")
    cmd_restart_k3s = f"echo '{sudo_pass}' | sudo -S systemctl restart k3s 2>/dev/null || echo '{sudo_pass}' | sudo -S systemctl restart k3s-agent"
    run_ssh_cmd(ip, cmd_restart_k3s, stream_output=True)

def setup_nvidia_device_plugin(head_ip, sudo_pass):
    print("\n>>> Deploying NVIDIA Device Plugin...")
    # This daemonset advertises the GPU resources to Kubernetes
    # Using specific commit hash provided by user
    plugin_url = "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/4f1de00eb6eb148dcd90cf85e9ce0fc830b43651/deployments/static/nvidia-device-plugin.yml"
    cmd = f"echo '{sudo_pass}' | sudo -S kubectl apply -f {plugin_url}"
    run_ssh_cmd(head_ip, cmd, stream_output=True)

def main():
    parser = argparse.ArgumentParser(description="Start K3s Cluster for Kubeflow")
    parser.add_argument("-y", "--yes", action="store_true", help="Auto-confirm node list")
    args = parser.parse_args()

    nodes = discover_nodes(auto_confirm=args.yes)
    if not nodes:
        print("No nodes found.")
        sys.exit(1)

    # Prompt for sudo password once
    print("\nNOTE: K3s installation requires root privileges (sudo) on remote nodes.")
    sudo_pass = getpass.getpass("Enter remote sudo password for user 'fabian': ")

    head_ip = nodes[0]
    worker_ips = nodes[1:]

    print(f"\nHead Node: {head_ip}")
    print(f"Worker Nodes: {worker_ips}")
    print("------------------------------------------------")
    print("NOTE: Using K3s instead of Minikube to support multi-node clustering.")
    print("      This will install K3s on all nodes and join them into a single cluster.")
    print("      It effectively replaces the 'minikube' requirement with a production-ready")
    print("      multi-node equivalent compatible with 'run_node.sh' style hardware.")
    print("------------------------------------------------")
    
    # 0. Build Images (Parallel)
    print("\n>>> Building Docker Images on all nodes...")
    build_threads = []
    for ip in nodes:
        t = threading.Thread(target=build_node_image, args=(ip,))
        t.start()
        build_threads.append(t)
    for t in build_threads:
        t.join()
    
    # 1. Install Head
    token = install_k3s_head(head_ip, sudo_pass)
    print(f"Token acquired: {token[:10]}...")

    # 2. Install Workers (Parallel)
    threads = []
    for ip in worker_ips:
        t = threading.Thread(target=install_k3s_worker, args=(ip, head_ip, token, sudo_pass))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()
    
    # 3. Configure NVIDIA Runtime (Parallel)
    print("\n>>> Configuring NVIDIA Runtime on all nodes...")
    gpu_threads = []
    for ip in nodes:
        t = threading.Thread(target=configure_nvidia_k3s, args=(ip, sudo_pass))
        t.start()
        gpu_threads.append(t)
    for t in gpu_threads:
        t.join()

    # 4. Setup NVIDIA Device Plugin (on Head)
    setup_nvidia_device_plugin(head_ip, sudo_pass)

    print("\n>>> Cluster Nodes:")
    run_ssh_cmd(head_ip, f"echo '{sudo_pass}' | sudo -S kubectl get nodes -o wide", stream_output=True)

    # 5. Check GPU Status
    check_gpu_status(nodes, sudo_pass)

    # 6. Install Kubeflow Training Operator
    setup_kubeflow_training_operator(head_ip, sudo_pass)
    
    # 7. Save Kubeconfig locally for the user
    print(f"\n>>> Fetching kubeconfig from {head_ip}...")
    local_kubeconfig = os.path.expanduser("~/.kube/config_k3s_custom")
    
    # We need to read the file, replace localhost/127.0.0.1 with the actual Head IP
    res = run_ssh_cmd(head_ip, f"echo '{sudo_pass}' | sudo -S cat /etc/rancher/k3s/k3s.yaml", check=True)
    config_content = res.stdout.replace("127.0.0.1", head_ip).replace("localhost", head_ip)
    
    # Modify for "static"/portable usage: skip TLS verify
    # This allows the config to work over SSH tunnels/different IPs without cert errors.
    new_lines = []
    for line in config_content.splitlines():
        if "certificate-authority-data:" in line:
            # Inject insecure-skip-tls-verify instead of the CA data
            indent = line.split("certificate-authority-data:")[0]
            new_lines.append(f"{indent}insecure-skip-tls-verify: true")
        else:
            new_lines.append(line)
    
    config_content = "\n".join(new_lines)
    
    os.makedirs(os.path.dirname(local_kubeconfig), exist_ok=True)
    with open(local_kubeconfig, "w") as f:
        f.write(config_content)
        
    print(f"Kubeconfig saved to: {local_kubeconfig}")
    print(f"To use: export KUBECONFIG={local_kubeconfig}")

def check_gpu_status(nodes, sudo_pass):
    print("\n>>> Checking GPU Status...")
    
    head_ip = nodes[0]
    
    # 1. Check Device Plugin Pods
    print(f"[{head_ip}] Checking NVIDIA Device Plugin Pods...")
    cmd_pods = f"echo '{sudo_pass}' | sudo -S kubectl get pods -n kube-system -l app=nvidia-device-plugin-daemonset"
    run_ssh_cmd(head_ip, cmd_pods, stream_output=True)
    
    # 2. Check Device Plugin Logs (Head)
    print(f"[{head_ip}] Checking Device Plugin Logs (Head)...")
    cmd_logs = f"echo '{sudo_pass}' | sudo -S kubectl logs -n kube-system -l app=nvidia-device-plugin-daemonset --tail=20"
    run_ssh_cmd(head_ip, cmd_logs, stream_output=True)
    
    # 3. Check Node Config (Head)
    print(f"[{head_ip}] Checking Docker config...")
    cmd_conf = f"echo '{sudo_pass}' | sudo -S cat /etc/docker/daemon.json"
    run_ssh_cmd(head_ip, cmd_conf, stream_output=True)

    # 4. Check K3s Logs for NVIDIA runtime (Head)
    print(f"[{head_ip}] Checking K3s logs for 'nvidia'...")
    # grep returns exit code 1 if no lines match, so we add '|| true' to prevent the script from crashing
    cmd_k3s_log = f"echo '{sudo_pass}' | sudo -S journalctl -u k3s -n 100 --no-pager | grep -i nvidia || true"
    run_ssh_cmd(head_ip, cmd_k3s_log, stream_output=True)

    print("\n>>> Node Capacity:")
    cmd_cap = f"echo '{sudo_pass}' | sudo -S kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.'nvidia\.com/gpu'"
    run_ssh_cmd(head_ip, cmd_cap, stream_output=True)

if __name__ == "__main__":
    main()
