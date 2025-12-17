#!/usr/bin/env python3
import ray
import os
import subprocess
import socket
import ctypes
import fcntl

# Configuration
NCCL_TESTS_REPO = "https://github.com/NVIDIA/nccl-tests.git"
NCCL_TESTS_DIR = os.path.abspath("nccl-tests")
BUILD_DIR = os.path.join(NCCL_TESTS_DIR, "build")
ALL_GATHER_PERF_BIN = os.path.join(BUILD_DIR, "all_gather_perf")


def get_nccl_unique_id_hex():
    """
    Generates a NCCL unique ID using libnccl and returns it as a hex string.
    """
    try:
        libnames = ["libnccl.so", "libnccl.so.2"]
        libnccl = None
        for name in libnames:
            try:
                libnccl = ctypes.CDLL(name)
                break
            except OSError:
                continue

        if not libnccl:
            raise RuntimeError("Could not load libnccl.so")

        # ncclGetUniqueId(ncclUniqueId* uniqueId)
        # ncclUniqueId is 128 bytes
        class NcclUniqueId(ctypes.Structure):
            _fields_ = [("internal", ctypes.c_char * 128)]

        unique_id = NcclUniqueId()
        res = libnccl.ncclGetUniqueId(ctypes.byref(unique_id))
        if res != 0:
            raise RuntimeError(f"ncclGetUniqueId failed with code {res}")

        # Convert to hex string
        return ctypes.string_at(ctypes.byref(unique_id), 128).hex()
    except Exception as e:
        print(f"Error generating NCCL ID: {e}")
        # Valid fallback if libnccl static: empty string and let it fail?
        # Or return None and rely on manual workaround?
        raise


def prepare_nccl_tests():
    """Clones, patches, and builds nccl-tests locally."""
    if not os.path.exists(NCCL_TESTS_DIR):
        print("Cloning nccl-tests...")
        subprocess.check_call(["git", "clone", NCCL_TESTS_REPO])

    common_cu_path = os.path.join(NCCL_TESTS_DIR, "src/common.cu")
    with open(common_cu_path, "r") as f:
        content = f.read()

    # Match run() function specifically
    run_func_sig = "testResult_t run() {"
    parts = content.split(run_func_sig)

    if len(parts) < 2:
        print("Warning: Could not split on run() signature. Check source file.")
        # Fallback or error?
        # Trying looser match?
        pass
    else:
        # 0. Insert headers
        if "#include <stdlib.h>" not in parts[0]:
            parts[0] = "#include <stdlib.h>\n" + parts[0]

        # 1. Insert Env Rank Parsing in run() body (parts[1])
        anchor1 = "getHostName(hostname, 1024);"
        check_proc = """
  // Ray/Env Support
  if (const char* env_rank = getenv("RANK")) proc = atoi(env_rank);
  if (const char* env_size = getenv("WORLD_SIZE")) totalProcs = atoi(env_size);
  // Simple local rank inference (assuming 1 proc per GPU or handled externally)
  if (const char* env_local_rank = getenv("LOCAL_RANK")) localRank = atoi(env_local_rank);
  
  ncclProcs = totalProcs;
  ncclProc = proc;
        """

        if anchor1 in parts[1]:
            parts[1] = parts[1].replace(anchor1, anchor1 + "\n" + check_proc, 1)
        else:
            print("Warning: Could not find anchor1 in run() body.")

        # 2. Insert ID Injection
        anchor2 = "ncclUniqueId ncclId;"
        anchor3 = "if (ncclProc == 0) {"

        if anchor2 in parts[1] and anchor3 in parts[1]:
            # Add Declaration
            parts[1] = parts[1].replace(
                anchor2, anchor2 + '\n const char* env_id = getenv("NCCL_TESTS_ID");', 1
            )

            # Add Branch
            injection_code = """
  if (env_id) {
      memset(&ncclId, 0, sizeof(ncclId));
      for (int i=0; i<128; i++) {
          unsigned int byte;
          sscanf(env_id + 2*i, "%02x", &byte);
          ncclId.internal[i] = (char)byte;
      }
  } else if (ncclProc == 0) {
             """
            parts[1] = parts[1].replace(anchor3, injection_code, 1)
        else:
            print("Warning: Could not find anchors for ID injection.")

        content = run_func_sig.join(parts)

    with open(common_cu_path, "w") as f:
        f.write(content)

    # Build
    # Always rebuild if we just patched or if binary missing?
    # Make handles dependency checking.
    print("Building nccl-tests (MPI=0)...")
    subprocess.check_call(["make", "MPI=0", "-j", "src.build"], cwd=NCCL_TESTS_DIR)


@ray.remote(num_gpus=1)
def run_worker(rank, world_size, local_rank, nccl_id_hex):
    # Set Environment
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["NCCL_TESTS_ID"] = nccl_id_hex

    hostname = socket.gethostname()
    bin_path = os.path.abspath(ALL_GATHER_PERF_BIN)

    # Lock for concurrent tasks on same node
    lock_file = "/tmp/nccl_tests_build.lock"
    with open(lock_file, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            # We re-check existence and also patch state?
            # prepare_nccl_tests handles patch idempotency.
            # But we only want to build if missing or if patch needed.
            # Let's just call prepare_nccl_tests(). It has file IO but shouldn't be too slow.
            # Efficiency: only called at start.
            prepare_nccl_tests()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    if not os.path.exists(bin_path):
        return f"Error: Binary not found on {hostname} after build attempt."

    cmd = [
        bin_path,
        "-b",
        "1G",
        "-e",
        "1G",
        "-f",
        "2",
        "-g",
        "1",
        "-w",
        "5",
        "-n",
        "10",
    ]

    print(f"[Rank {rank}] Executing on {hostname}: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ
    )

    if result.returncode != 0:
        return f"Rank {rank} failed (RC={result.returncode}): {result.stderr}\nStdout: {result.stdout}"

    return result.stdout


def main():
    print("Connecting to Ray...")
    ray.init(address="auto")

    print("Generatiing NCCL ID...")
    nccl_id_hex = get_nccl_unique_id_hex()
    print(f"ID: {nccl_id_hex}")

    print("Discovering GPU nodes...")
    nodes = ray.nodes()
    gpu_nodes = [n for n in nodes if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
    unique_gpu_ips = sorted(list(set([n["NodeManagerAddress"] for n in gpu_nodes])))

    assignments = []
    rank = 0
    for ip in unique_gpu_ips:
        node = next(n for n in gpu_nodes if n["NodeManagerAddress"] == ip)
        gpu_count = int(node["Resources"].get("GPU", 0))
        for lr in range(gpu_count):
            assignments.append(
                {"node_id": node["NodeID"], "ip": ip, "rank": rank, "local_rank": lr}
            )
            rank += 1

    world_size = rank  # total ranks
    print(f"World Size: {world_size}")

    futures = []
    for a in assignments:
        print(f"Scheduling Rank {a['rank']} on {a['ip']}")
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        futures.append(
            run_worker.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=a["node_id"], soft=False
                )
            ).remote(a["rank"], world_size, a["local_rank"], nccl_id_hex)
        )

    print("Tasks submitted. Waiting for results...")
    results = ray.get(futures)

    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    for i, res in enumerate(results):
        print(f"Rank {i} Output:")
        print(res)
        print("-" * 40)


if __name__ == "__main__":
    main()
