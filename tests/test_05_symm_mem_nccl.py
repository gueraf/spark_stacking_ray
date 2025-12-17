#!/usr/bin/env python3
import ray
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import os
import time
import socket


@ray.remote(num_gpus=1)
def run_worker(rank, world_size, master_addr, master_port):
    import os
    import socket

    print(f"[Rank {rank} @ {socket.gethostname()}] Initializing...")

    # Set Rendezvous Info
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Initialize Process Group
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
    except Exception as e:
        return f"Rank {rank} failed to init dist: {e}"

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    # Set Backend to NCCL
    print(f"[Rank {rank}] Setting symm_mem backend to NCCL...")
    try:
        symm_mem.set_backend("NCCL")
        # "enable_symm_mem_for_group" takes a string group name.
        # The previous error "no group info associated with the group name 0" suggests "0" is the target.
        symm_mem.enable_symm_mem_for_group("0")
    except Exception as e:
        return f"Rank {rank} Failed to set backend/enable group: {e}"

    # Allocate Data
    try:
        print(f"[Rank {rank}] Allocating symm_mem tensor...")
        # Create input tensor (float32 = 4 bytes)
        # Small size for functional verification
        num_elements = 1024 * 1024  # 1M elements, 4MB

        # Note: symm_mem.empty might use the backend set globally
        t = symm_mem.empty(num_elements, dtype=torch.float32, device=device)

        print(f"[Rank {rank}] Rendezvous...")
        hdl = symm_mem.rendezvous(t, dist.group.WORLD)

        print(f"[Rank {rank}] Rendezvous complete. Handle obtained.")

        # Verify handle properties just to ensure it's not empty/broken
        # hdl.buffer_ptrs should be available
        if hdl.buffer_ptrs is None:
            raise RuntimeError("hdl.buffer_ptrs is None")

        print(f"[Rank {rank}] buffer_ptrs: {hdl.buffer_ptrs}")

    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"Rank {rank} SymmMem Error: {e}"

    # Cleanup
    dist.destroy_process_group()
    return "Success"


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
        print("Warning: Need at least 2 nodes.")
        # Try to run anyway if possible, but SymmMem usually needs peers for interesting stuff.

    master_addr = unique_gpu_ips[0]
    master_port = 29506

    futures = []
    for rank, ip in enumerate(unique_gpu_ips):
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        node_id = next(n["NodeID"] for n in gpu_nodes if n["NodeManagerAddress"] == ip)

        print(f"Scheduling Rank {rank} on {ip}")
        futures.append(
            run_worker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=node_id, soft=False
                )
            ).remote(rank, world_size, master_addr, master_port)
        )

    print("Tasks submitted. Waiting for results...")
    results = ray.get(futures)

    print("Results:", results)

    for r in results:
        if r != "Success":
            print("Test FAILED")
            exit(1)

    print("Test PASSED")


if __name__ == "__main__":
    main()
