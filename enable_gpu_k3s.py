#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import threading
import argparse
import getpass

# Reusing helper functions from start_kubeflow.py

def run_ssh_cmd(ip, cmd, check=True, stream_output=False, env=None):
    ssh_cmd = ["ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", ip, cmd]
    if stream_output:
        return subprocess.run(ssh_cmd, check=check, env=env)
    else:
        return subprocess.run(ssh_cmd, check=check, capture_output=True, text=True, env=env)

def discover_nodes(auto_confirm=False):
    print("Discovering nodes (using Avahi)...")
    # ... (same logic as before, simplified for brevity as we know it works)
    try:
         out = subprocess.check_output("avahi-browse -p -r -f -t _ssh._tcp", shell=True).decode()
         ips = set()
         for line in out.splitlines():
             if not line.startswith("="): continue
             parts = line.split(";")
             if len(parts) > 7:
                 ip = parts[7]
                 if ":" in ip: continue
                 if ip.startswith("10."): # Simple filter for your network
                     ips.add(ip)
         nodes = sorted(list(ips))
    except:
        nodes = ["127.0.0.1"]
    
    print(f"Found nodes: {nodes}")
    if auto_confirm: return nodes
    
    c = input("Confirm node list (Y/n): ").strip().lower()
    if c == '' or c == 'y': return nodes
    sys.exit(0)

def configure_nvidia_k3s(ip, sudo_pass):
    print(f"[{ip}] Configuring NVIDIA support for K3s...")
    
    commands = [
        # 1. Install toolkit if missing
        "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null",
        "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null",
        "sudo apt-get update -qq",
        "sudo apt-get install -y -qq nvidia-container-toolkit",
        
        # 2. Configure for K3s (containerd)
        # We explicitly generate the config.toml for k3s
        "sudo mkdir -p /var/lib/rancher/k3s/agent/etc/containerd",
        """sudo bash -c 'cat <<EOF > /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl\n[plugins.opt]\n  path = \"/var/lib/rancher/k3s/agent/containerd\"\n  [plugins.cri.containerd]\n    snapshotter = \"overlayfs\"\n    [plugins.cri.containerd.runtimes.runc]\n      runtime_type = \"io.containerd.runc.v2\"\n    [plugins.cri.containerd.runtimes.nvidia]\n      runtime_type = \"io.containerd.runc.v2\"\n      [plugins.cri.containerd.runtimes.nvidia.options]\n        BinaryName = \"/usr/bin/nvidia-container-runtime\"\nEOF'""",
    ]
    
    full_cmd = " && ".join(commands)
    # Inject password
    full_cmd = full_cmd.replace("sudo", f"echo '{sudo_pass}' | sudo -S")
    
    try:
        run_ssh_cmd(ip, full_cmd, stream_output=True)
    except Exception as e:
        print(f"[{ip}] Error configuring NVIDIA: {e}")

    # 3. Restart K3s
    print(f"[{ip}] Restarting K3s...")
    restart_cmd = f"echo '{sudo_pass}' | sudo -S systemctl restart k3s 2>/dev/null || echo '{sudo_pass}' | sudo -S systemctl restart k3s-agent"
    run_ssh_cmd(ip, restart_cmd, stream_output=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-y", "--yes", action="store_true")
    args = parser.parse_args()
    
    nodes = discover_nodes(args.yes)
    sudo_pass = getpass.getpass("Enter remote sudo password: ")
    
    threads = []
    for ip in nodes:
        t = threading.Thread(target=configure_nvidia_k3s, args=(ip, sudo_pass))
        t.start()
        threads.append(t)
    for t in threads: t.join()

    print("\n>>> Deploying NVIDIA Device Plugin...")
    head_ip = nodes[0]
    plugin_url = "https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml"
    cmd = f"echo '{sudo_pass}' | sudo -S kubectl apply -f {plugin_url}"
    run_ssh_cmd(head_ip, cmd, stream_output=True)
    
    print("\nDone. Please wait a moment for nodes to register GPUs.")
    print("Check with: kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.allocatable.'nvidia.com/gpu'")

if __name__ == "__main__":
    main()
