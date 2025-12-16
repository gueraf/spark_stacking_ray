#!/usr/bin/env python3
import os
import sys
import subprocess
import time

INFO_FILE = "ray_info.txt"
RUN_SCRIPT = "run_node.sh"
DOCKERFILE = "Dockerfile"
REMOTE_BUILD_DIR = "/tmp/ray_build"
DOCKER_IMAGE = "ray_nccl_node" # Local built image

def run_ssh_cmd(ip, cmd, check=True, stream_output=False):
    ssh_cmd = ["ssh", "-T", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", ip, cmd]
    
    if stream_output:
        return subprocess.run(ssh_cmd, check=check)
    else:
        return subprocess.run(ssh_cmd, check=check, capture_output=True, text=True)

def scp_file(ip, local_path, remote_path):
    cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", local_path, f"{ip}:{remote_path}"]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_node_details(ip):
    print(f"[{ip}] Probing interfaces...")
    # Try to find active IB/RoCE interface
    # Output format: "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"
    try:
        # We look for "Up" and exclude "p2p" explicitly in the grep chain to match user pref
        res = run_ssh_cmd(ip, "ibdev2netdev | grep 'Up' | grep -v 'p2p' | head -n 1")
        line = res.stdout.strip()
        
        # If strict filter fails, fallback to any Up interface (maybe p2p is the only one?)
        # But user requested "don't use the p2p one". So we stick to strict.
        
        if line:
            # Parse: "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"
            parts = line.split("==>")
            if len(parts) == 2:
                # Left side: "rocep1s0f1 port 1 " -> "rocep1s0f1"
                ib_dev = parts[0].split()[0]
                # Right side: " enp1s0f1np1 (Up)" -> "enp1s0f1np1"
                eth_dev = parts[1].strip().split()[0]
                return eth_dev, ib_dev
    except:
        pass
    
    # Fallback
    print(f"[{ip}] Warning: Could not detect IB interface via ibdev2netdev. Using eth0/mlx5_0.")
    return "eth0", "mlx5_0" 

def discover_nodes():
    print("Discovering nodes...")
    
    # 1. Identify Preferred Interface (Local)
    preferred_iface = None
    try:
        # Check local ibdev2netdev for "Up" and non-p2p
        out = subprocess.check_output("ibdev2netdev | grep 'Up' | grep -v 'p2p'", shell=True).decode()
        for line in out.splitlines():
            # "rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)"
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
         
         # 3. Filter by Preferred Interface
         for line in out.splitlines():
             # =;enp1s0f1np1;IPv4;yang-zrh...;10.10.20.2;22;
             if not line.startswith("="): continue
             parts = line.split(";")
             if len(parts) > 7:
                 iface = parts[1]
                 ip = parts[7]
                 
                 # Only accept IPv4
                 if ":" in ip: continue
                 
                 # If we found a preferred interface, filter by it
                 if preferred_iface and iface != preferred_iface:
                     continue
                 
                 ips.add(ip)
         
         nodes = sorted(list(ips))
    except:
        nodes = ["127.0.0.1"]

    print(f"Found nodes: {nodes}")
    while True:
        c = input("Confirm node list (Y/n/manual): ").strip().lower()
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

import threading

def provision_node_thread(node, i, nodes):
    role = "head" if i == 0 else "node"
    head_ip = nodes[0]
    print(f"\n[Thread {node}] Setting up {role.upper()}...")
    
    # 1. Get Details
    eth_if, ib_if = get_node_details(node)
    print(f"    [{node}] Interface: {eth_if}, IB: {ib_if}")
    
    # 2. Build Docker Image (Ephemeral)
    print(f"    [{node}] Preparing build context...")
    run_ssh_cmd(node, f"mkdir -p {REMOTE_BUILD_DIR}")
    scp_file(node, DOCKERFILE, f"{REMOTE_BUILD_DIR}/Dockerfile")
    scp_file(node, RUN_SCRIPT, f"{REMOTE_BUILD_DIR}/run_node.sh")
    
    print(f"    [{node}] Building Docker image (this may take a while)...")
    res = run_ssh_cmd(node, f"cd {REMOTE_BUILD_DIR} && docker build -t {DOCKER_IMAGE} .", stream_output=True)
    if res.returncode != 0:
        print(f"    [{node}] Build FAILED.")
        # We can't exit main thread easily, just return?
        return
    
    # 3. Stop existing containers
    print(f"    [{node}] Stopping old containers...")
    run_ssh_cmd(node, "docker rm -f ray_cluster_node 2>/dev/null || true")
    
    # 4. Launch Docker
    docker_cmd = [
        "docker", "run", "-d",
        "--name", "ray_cluster_node",
        "--net=host",
        "--ipc=host",
        "--privileged",
        "--gpus=all",
        "--shm-size=10.24gb",
        "--restart=always",
        DOCKER_IMAGE,
        "--role", role,
        "--host-ip", node,
        "--eth-if", eth_if,
        "--ib-if", ib_if
    ]
    
    if role == "node":
        docker_cmd.extend(["--head-ip", head_ip])
        
    cmd_str = " ".join(docker_cmd)
    print(f"    [{node}] Launching container...")
    print(f"    [{node}] [DEBUG] Running: {cmd_str}")
    
    res = run_ssh_cmd(node, cmd_str, stream_output=True)
    
    if res.returncode != 0:
        print(f"    [{node}] Launch FAILED with exit code {res.returncode}")
    else:
        print(f"    [{node}] Success.")

def start_cluster():
    nodes = discover_nodes()
    head_ip = nodes[0]
    
    print("\n>>> Preparing Cluster (Parallel)...")
    
    threads = []
    for i, node in enumerate(nodes):
        t = threading.Thread(target=provision_node_thread, args=(node, i, nodes))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()

    print("\nCluster Launched!")
    print(f"Head Node: {head_ip}")
    print(f"Ray Dashboard: http://{head_ip}:8265")
    print("To check logs: ssh <ip> 'docker logs -f ray_cluster_node'")
    print("Use 'python test.py' to run workloads.")

if __name__ == "__main__":
    start_cluster()
