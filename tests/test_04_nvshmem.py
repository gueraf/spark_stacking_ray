#!/usr/bin/env python3
import ray
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import os
import time


# Imports handled inside workers or not needed
try:
    import torch.distributed._symmetric_memory._nvshmem_triton
except ImportError:
    pass  # Check happens inside worker or kernel module if needed


# Configuration
DATA_SIZE_BYTES = 1 * 1024**3  # 1 GB per transfer to be safe
NUM_ITERS = 100
WARMUP_ITERS = 10


@ray.remote(num_gpus=1)
def run_worker(rank, world_size, master_addr, master_port):
    # Import inside worker to avoid pickling JIT object on driver
    try:
        from tests.nvshmem_kernels import my_put_kernel
    except ImportError:
        try:
            from nvshmem_kernels import my_put_kernel
        except ImportError:
            raise ImportError("Could not import nvshmem_kernels on worker")

    # Set Rendezvous Info
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)

    # Initialize Process Group
    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    print(f"[Rank {rank}] SymmMem init...")
    # Create symmetric input tensor
    # float32 = 4 bytes
    num_elements = DATA_SIZE_BYTES // 4

    # Allocate symmetric memory
    # Note: symm_mem.empty requires device kwarg or uses current device? Docs say: empty(size, device=...)
    try:
        src_tensor = symm_mem.empty(num_elements, dtype=torch.float32, device=device)
        # We also need a handle to establish mapping, though nvshmem might handle it via internal heap
        # providing we follow setup. symm_mem.rendezvous does the handshake.
        hdl = symm_mem.rendezvous(src_tensor, dist.group.WORLD)
    except Exception as e:
        print(f"[Rank {rank}] Error allocating symm_mem: {e}")
        return f"Rank {rank} SymmMem Error: {e}"

    # Initialize data
    src_tensor.fill_(rank + 1.0)

    # We need a destination address on the peer.
    # In NVSHMEM, usually you access the symm heap at the symmetric offset.
    # PyTorch SymmMem `hdl.buffer_ptrs` gives pointers to the buffer on all ranks.
    # We can use this to get the specific pointer to pass to put?
    # Actually NVSHMEM usually takes a symmetric address (local pointer) + PE,
    # OR a global pointer.
    # The doc says: "nvshmem.put(dest, src, nelems, pe)"
    # If `dest` is a pointer, it should probably be the symmetric address (i.e. src_tensor.data_ptr())?
    # Or if we want to write to peer's buffer implementation:
    # The doc says "primitives that are slightly higher level than pointer access, such as put and get".
    # And "initiate a cross-node put command within the kernel".

    # Let's look at `hdl.buffer_ptrs`.
    # But wait, if we use NVSHMEM, we might rely on NVSHMEM's symmetric allocation.
    # `symm_mem.empty` backs it by a commercially compatible allocator (likely checking if nvshmem is available).

    # For `nvshmem.put(dest, src, nelems, pe)`:
    # dest: Remote address? Or symmetric address?
    # Standard NVSHMEM put takes: void *dest, const void *source, size_t nelems, int pe.
    # "dest: Symmetric address of the destination data object." (NVSHMEM docs)
    # So `dest` should be the address of the tensor on the LOCAL node (which represents the symmetric address).
    # `src` is the local source.
    # `pe` is the target rank.

    dest_ptr = src_tensor.data_ptr()
    src_ptr = (
        src_tensor.data_ptr()
    )  # We can read from our own buffer, or another freq buffer.
    # Let's allocate a separate local buffer for source to avoid confusion/overlap if needed,
    # but for benchmark 'put' usually sends local data to remote.
    # Let's create a regular tensor as source to be clean.
    local_src = torch.ones(num_elements, dtype=torch.float32, device=device) * (
        rank + 10.0
    )
    src_ptr = local_src.data_ptr()

    # Peer to write to: (rank + 1) % world_size
    target_rank = (rank + 1) % world_size

    # Clean up before bench
    torch.cuda.synchronize()
    dist.barrier()

    # Kernel Launch Config
    # We launch 1 instance to issue the put command?
    # Or multiple? NVSHMEM operations are typically device-initiated.
    # The doc example usage isn't fully detailed on grid size.
    # Let's try grid=(1,) for simply issuing the command.
    grid = (1,)

    print(f"[Rank {rank}] Starting Warmup...")
    for _ in range(WARMUP_ITERS):
        my_put_kernel[grid](
            dest_ptr, src_ptr, num_elements, target_rank, BLOCK_SIZE=1024
        )

    torch.cuda.synchronize()
    dist.barrier()

    print(f"[Rank {rank}] Starting Benchmark...")
    start_time = time.time()
    for _ in range(NUM_ITERS):
        my_put_kernel[grid](
            dest_ptr, src_ptr, num_elements, target_rank, BLOCK_SIZE=1024
        )

    torch.cuda.synchronize()  # Wait for kernel to finish
    # Note: NVSHMEM quiet might be needed?
    # But torch.cuda.synchronize() waits for the stream.
    # The put operation is ordered on the stream?
    # Usually yes for triton kernels.

    end_time = time.time()
    dist.barrier()

    total_time_us = (end_time - start_time) * 1e6
    avg_time_us = total_time_us / NUM_ITERS
    size_gb = DATA_SIZE_BYTES / 1e9
    alg_bw = size_gb / (avg_time_us / 1e6)

    # Teardown
    dist.destroy_process_group()

    results = {
        "rank": rank,
        "size_bytes": DATA_SIZE_BYTES,
        "avg_time_us": avg_time_us,
        "alg_bw": alg_bw,
    }
    return results


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
        print(
            "Warning: Need at least 2 nodes/GPUs for meaningful nvshmem test (cross-rank)."
        )
        # Proceeding anyway might work for loopback or single node multiple GPU if set up differently,
        # but the request implies 'nvshmem' often for scale out.
        # If we have 1 node with multiple GPUs, unique_gpu_ips might be 1.
        # Let's count total GPUs.
        total_gpus = sum(n["Resources"].get("GPU", 0) for n in gpu_nodes)
        if total_gpus < 2:
            print("Error: Less than 2 GPUs found. Aborting.")
            return

        # If 1 node multiple GPUs, we can spawn multi-rank.
        if world_size == 1 and total_gpus >= 2:
            print("Running on single node with multiple GPUs.")
            # We need to launch multiple workers on same node.
            # Adjust strategy.
            world_size = int(total_gpus)
            gpu_ips_schedule = [unique_gpu_ips[0]] * world_size
        else:
            # One GPU per node assumption or 1 rank per node for simplicity?
            # Existing test_03 assumes 1 rank per node (unique_gpu_ips).
            # Let's stick to unique_gpu_ips logic from test_03 but handle single node multi-gpu if needed.
            # Actually test_03 logic:
            # unique_gpu_ips = sorted(list(set([n["NodeManagerAddress"] for n in gpu_nodes])))
            # world_size = len(unique_gpu_ips)
            # This implies 1 rank per node.
            pass

    master_addr = unique_gpu_ips[0]
    master_port = 29505

    print(f"Master: {master_addr}:{master_port}")

    futures = []
    # If using test_03 logic (1 rank per node):
    used_ips = unique_gpu_ips

    for rank, ip in enumerate(used_ips):
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        # Find node id for this IP
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

    print("\n" + "=" * 80)
    print(f"{'Rank':>5} {'size (B)':>15} {'time (us)':>10} {'BW (GB/s)':>15}")
    print("=" * 80)
    for r in results:
        if isinstance(r, str):
            print(f"Error: {r}")
        elif isinstance(r, dict):
            print(
                f"{r['rank']:5d} {r['size_bytes']:15d} {r['avg_time_us']:10.0f} {r['alg_bw']:15.2f}"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()
