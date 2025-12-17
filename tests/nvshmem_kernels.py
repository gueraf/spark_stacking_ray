import triton
import triton.language as tl

try:
    import torch.distributed._symmetric_memory._nvshmem_triton as nvshmem
    from torch.distributed._symmetric_memory._nvshmem_triton import requires_nvshmem
except ImportError:
    # Fallback or dummy if not available during lint/check, but expected at runtime
    pass


@requires_nvshmem
@triton.jit
def my_put_kernel(
    dest_ptr,  # Pointer to destination buffer (remote)
    src_ptr,  # Pointer to source buffer (local)
    nelems,  # Number of elements
    pe,  # Target PE (Processing Element / Rank)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    # Simple single-block or mapped behavior.
    # We rely on nvshmem.put to handle transfer.
    nvshmem.put(dest_ptr, src_ptr, nelems, pe)
