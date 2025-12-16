
import ray
import torch
import torch.distributed as dist
import os
import time
import socket

# Configuration
DATA_SIZE_BYTES = 16 * 1024**3  # 16 GB
NUM_ITERS = 20
WARMUP_ITERS = 5

@ray.remote(num_gpus=1)
def run_worker(rank, world_size, master_addr, master_port):
    # Set Rendezvous Info
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    
    # Print inherited environment variables for verification
    print(f"\n[Rank {rank} @ {socket.gethostname()}] PyTorch Version: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"[Rank {rank} @ {socket.gethostname()}] Network Environment:")
    keys = [
        "NCCL_SOCKET_IFNAME", 
        "UCX_NET_DEVICES", 
        "NCCL_IB_HCA", 
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG",
        "LD_LIBRARY_PATH"
    ]
    for key in keys:
        print(f"  {key}: {os.environ.get(key, 'Not Set')}")

    # Check loaded NCCL library
    try:
        import ctypes.util
        lib = ctypes.util.find_library("nccl")
        print(f"  [Check] found libnccl via ctypes: {lib}")
        
        # More robust check: inspect /proc/self/maps for loaded .so
        print("  [Check] Loaded libnccl.so paths in process:")
        with open("/proc/self/maps") as f:
            for line in f:
                if "libnccl.so" in line:
                    path = line.split()[-1]
                    print(f"    - {path}")
    except Exception as e:
        print(f"  [Check] Failed to check loaded libraries: {e}")

    # Initialize Process Group
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
    except Exception as e:
        return f"Rank {rank} failed to init: {e}"

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    # Allocate Data
    try:
        # Create input tensor (float32 = 4 bytes)
        num_elements = DATA_SIZE_BYTES // 4
        input_tensor = torch.ones(num_elements, dtype=torch.float32, device=device)
        output_list = [torch.zeros_like(input_tensor) for _ in range(world_size)]
    except RuntimeError as e:
        return f"Rank {rank} OOM: {e}"

    # Warmup
    for _ in range(WARMUP_ITERS):
        dist.all_gather(output_list, input_tensor)
    torch.cuda.synchronize()

    # Measurement Loop
    start_time = time.time()
    for _ in range(NUM_ITERS):
        dist.all_gather(output_list, input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    total_time_us = (end_time - start_time) * 1e6
    avg_time_us = total_time_us / NUM_ITERS
    
    # Bandwidth Calculation
    size_gb = DATA_SIZE_BYTES / 1e9
    alg_bw = size_gb / (avg_time_us / 1e6)
    bus_bw = alg_bw * (world_size - 1) / world_size

    dist.destroy_process_group()

    if rank == 0:
        return {
            "size_bytes": DATA_SIZE_BYTES,
            "count_elements": num_elements,
            "avg_time_us": avg_time_us,
            "alg_bw": alg_bw,
            "bus_bw": bus_bw
        }
    return None

def main():
    print("Connecting to Ray...")
    ray.init(address="auto")

    print("Discovering GPU nodes...")
    nodes = ray.nodes()
    gpu_nodes = [n for n in nodes if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
    
    unique_gpu_ips = sorted(list(set([n["NodeManagerAddress"] for n in gpu_nodes])))
    world_size = len(unique_gpu_ips)
    print(f"Found {world_size} GPU nodes: {unique_gpu_ips}")
    
    if world_size < 2:
        print("Warning: Need at least 2 nodes for meaningful all_gather test.")

    master_addr = unique_gpu_ips[0]
    master_port = 29500

    print(f"Master: {master_addr}:{master_port}")

    futures = []
    for rank, ip in enumerate(unique_gpu_ips):
        # Schedule specifically on this node using NodeAffinity
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
        node_id = next(n["NodeID"] for n in gpu_nodes if n["NodeManagerAddress"] == ip)
        
        print(f"Scheduling Rank {rank} on {ip}")
        futures.append(
            run_worker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote(rank, world_size, master_addr, master_port)
        )

    print("Tasks submitted. Waiting for results...")
    results = ray.get(futures)
    
    stats = results[0]
    if isinstance(stats, str):
        print(f"Error in Rank 0: {stats}")
    elif stats:
        print("\n" + "="*80)
        print(f"{ 'size (B)':>15} { 'count':>15} { 'type':>8} { 'time (us)':>10} { 'algbw (GB/s)':>15} { 'busbw (GB/s)':>15}")
        print("="*80)
        print(f"{stats['size_bytes']:15d} {stats['count_elements']:15d} {'float':>8} {stats['avg_time_us']:10.0f} {stats['alg_bw']:15.2f} {stats['bus_bw']:15.2f}")
        print("="*80)
    else:
        for r in results:
            if isinstance(r, str):
                print(f"Error: {r}")

if __name__ == "__main__":
    main()
