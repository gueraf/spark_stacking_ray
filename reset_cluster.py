#!/usr/bin/env python3
import sys
import subprocess
import getpass
import argparse

# Reusing helper from start_kubeflow.py for consistency
def run_ssh_cmd(ip, cmd, check=True):
    ssh_cmd = ["ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", ip, cmd]
    return subprocess.run(ssh_cmd, check=check)

def discover_nodes():
    # ... Simplified discovery ...
    try:
         out = subprocess.check_output("avahi-browse -p -r -f -t _ssh._tcp", shell=True).decode()
         ips = set()
         for line in out.splitlines():
             if not line.startswith("="): continue
             parts = line.split(";")
             if len(parts) > 7:
                 ip = parts[7]
                 if ":" in ip: continue
                 if ip.startswith("10."): ips.add(ip)
         return sorted(list(ips))
    except:
        return ["127.0.0.1"]

def reset_node(ip, sudo_pass):
    print(f"[{ip}] Wiping K3s and Container State...")
    
    commands = [
        # 1. Uninstall K3s (Head or Agent)
        "/usr/local/bin/k3s-uninstall.sh || /usr/local/bin/k3s-agent-uninstall.sh || true",
        
        # 2. Kill any leftover processes
        "killall -9 k3s k3s-agent containerd kubelet 2>/dev/null || true",
        
        # 3. Clean up directories (CRITICAL for switching runtimes)
        "rm -rf /var/lib/rancher /etc/rancher /var/lib/kubelet /var/lib/cni /etc/cni /run/k3s",
        
        # 4. Clean up generated config files
        "rm -f /var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl",
        
        # 5. Prune Docker (Optional but good)
        "docker system prune -f"
    ]
    
    full_cmd = " && ".join(commands)
    # Inject password
    full_cmd = f"echo '{sudo_pass}' | sudo -S sh -c '{full_cmd}'"
    
    try:
        run_ssh_cmd(ip, full_cmd, check=False)
        print(f"[{ip}] Reset Complete.")
    except Exception as e:
        print(f"[{ip}] Reset Failed: {e}")

def main():
    print("WARNING: This will completely wipe K3s from all discovered nodes.")
    if input("Continue? (y/N): ").lower() != 'y':
        sys.exit(0)
        
    nodes = discover_nodes()
    print(f"Nodes: {nodes}")
    
    sudo_pass = getpass.getpass("Enter remote sudo password: ")
    
    for ip in nodes:
        reset_node(ip, sudo_pass)
        
if __name__ == "__main__":
    main()
