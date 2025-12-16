import ray
import subprocess
import socket
import sys

@ray.remote(num_gpus=1)
def check_gpu():
    hostname = socket.gethostname()
    try:
        output = subprocess.check_output("nvidia-smi -L", shell=True).decode()
        print(f"[{hostname}] nvidia-smi output:\n{output}")
        gpu_count = len(output.strip().split('\n'))
        return hostname, gpu_count, output
    except Exception as e:
        return hostname, 0, str(e)

def main():
    ray.init(address="auto")
    print("Connected to Ray.")
    
    nodes = ray.nodes()
    gpu_nodes = [n for n in nodes if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
    unique_gpu_ips = sorted(list(set([n["NodeManagerAddress"] for n in gpu_nodes])))
    
    print(f"Found {len(unique_gpu_ips)} GPU nodes: {unique_gpu_ips}")

    futures = []
    # We want to run one check per node
    # We can use NodeAffinity to ensure we hit every node
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    for ip in unique_gpu_ips:
        node_id = next(n["NodeID"] for n in gpu_nodes if n["NodeManagerAddress"] == ip)
        futures.append(
            check_gpu.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote()
        )

    results = ray.get(futures)
    
    all_passed = True
    for hostname, count, output in results:
        if count >= 1:
            print(f"PASS: Node {hostname} has {count} GPUs.")
        else:
            print(f"FAIL: Node {hostname} has {count} GPUs. Output: {output}")
            all_passed = False
            
    if all_passed:
        print("Test PASSED: All nodes have at least 1 GPU.")
    else:
        print("Test FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
