"""
Minimal Fused Mixture-of-Experts (MoE) implementation with a custom Triton kernel.

=== Background: Why a Custom Kernel? ===

In a Mixture of Experts layer, each token is routed to K out of E experts
(small MLPs). A naive implementation would loop over experts in Python:

    for expert in experts:
        find tokens assigned to this expert
        output[mask] += weight * expert(input[mask])

This launches E separate small GEMMs (matrix multiplications). GPUs are
optimized for large GEMMs, not many small ones — each kernel launch has
overhead and the GPU is underutilized between launches.

The Triton FusedMoE kernel solves this by treating ALL expert computations
as a single fused operation. The key idea:

1. PRE-SORT tokens by expert assignment so all tokens for expert E are
   contiguous in memory
2. Launch ONE kernel where each thread block knows which expert's weights
   to use (looked up from the sorted mapping)
3. Each thread block computes a tile of the output matrix — standard tiled
   GEMM, but with expert-aware weight selection

This gives us one large kernel launch instead of E small ones, with good
memory coalescing (tokens for the same expert are adjacent).

=== Expert Parallelism (EP) ===

When using multiple GPUs, experts are distributed across GPUs. Each GPU
holds only num_experts // num_gpus experts. The kernel handles this via
an expert_map tensor that maps global expert IDs to local IDs (or -1 for
non-local experts). Blocks assigned to expert_id == -1 simply write zeros
and return early — no wasted compute.

After the kernel, an all-reduce across GPUs sums the partial outputs
(each GPU contributed results for its local experts only).

=== CUDA Graph Compatibility ===

CUDA graphs record a sequence of GPU kernel launches once, then "replay"
them on future steps — eliminating CPU-side kernel launch overhead. This
is critical for decode (1 token per sequence) where the GPU work per step
is tiny and launch overhead dominates.

The challenge with MoE is that the token sorting step traditionally uses
operations that break CUDA graphs:

  1. `.item()` calls that force CPU-GPU synchronization
  2. Dynamic-size tensor allocations (size depends on routing results)
  3. `torch.repeat_interleave` with data-dependent output sizes

Our solution uses three techniques to make MoE graph-compatible:

  1. **Pre-allocated fixed-size buffers**: All sorting outputs are allocated
     once to worst-case maximum size. The Triton kernel early-exits for
     unused blocks via a GPU-resident count (no CPU involvement).

  2. **Triton sorting kernels**: A histogram kernel (counts pairs per expert
     via atomic adds) and a scatter kernel (writes pair indices into sorted
     positions via atomic slot reservation) replace the old argsort-based
     sorting. Both are O(n) and fully GPU-resident.

  3. **GPU-resident metadata**: `num_tokens_post_padded` stays as a GPU
     tensor (never pulled to CPU with `.item()`). The GEMM kernel reads it
     directly for early-exit decisions. `torch.searchsorted` replaces
     `repeat_interleave` for expert_ids, giving fixed output size.
"""

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Block size for the Triton fused MoE GEMM kernel's M dimension (tokens).
# This MUST match the BLOCK_SIZE_M used in invoke_fused_moe_kernel.
# The sorting step pads each expert's token count to a multiple of this.
MOE_BLOCK_SIZE_M = 64


# ===========================================================================
# Triton Sorting Kernels (CUDA-Graph-Safe)
# ===========================================================================
#
# These two small Triton kernels replace the Python-level token sorting
# that was incompatible with CUDA graphs. Together they sort token-expert
# pairs by expert in O(n) time using GPU atomics.
#
# Old approach (graph-BREAKING):
#   1. bincount → histogram       (OK but we replace for consistency)
#   2. .item() → CPU-GPU sync     ← BREAKS GRAPH
#   3. torch.full(dynamic_size)   ← BREAKS GRAPH (variable allocation)
#   4. repeat_interleave          ← BREAKS GRAPH (variable output size)
#   5. argsort + scatter          (O(n log n), replaced with O(n) atomics)
#
# New approach (graph-SAFE):
#   1. Triton histogram kernel    → count pairs per expert (GPU atomics)
#   2. cumsum on GPU              → expert offsets (standard CUDA op)
#   3. GPU tensor assignment      → total count stays on GPU (no .item())
#   4. searchsorted               → expert_ids with fixed output size
#   5. Triton scatter kernel      → fill sorted_token_ids (GPU atomics)
#   6. All buffers pre-allocated  → no dynamic allocation
#
# ===========================================================================


@triton.jit
def _moe_histogram_kernel(
    # --- Pointers ---
    topk_ids_ptr,              # [num_pairs] flattened expert assignments (int32)
    tokens_per_expert_ptr,     # [E] output histogram, MUST be pre-zeroed (int32)
    # --- Scalars ---
    num_pairs,                 # Total number of token-expert pairs (T * K)
    # --- Compile-time constants ---
    BLOCK_SIZE: tl.constexpr,  # Pairs processed per Triton program (e.g., 256)
):
    """
    Count how many token-expert pairs are assigned to each expert.

    Replaces `torch.bincount(flat_ids, minlength=num_experts)` with a
    Triton kernel that uses `tl.atomic_add` for CUDA-graph-safe counting.

    === How it works ===

    The grid has ceil(num_pairs / BLOCK_SIZE) programs. Each program
    processes BLOCK_SIZE consecutive pairs from the flattened topk_ids:

        Program 0: pairs [0, BLOCK_SIZE)
        Program 1: pairs [BLOCK_SIZE, 2*BLOCK_SIZE)
        ...

    For each pair, the kernel loads the expert ID and atomically increments
    that expert's counter in the output histogram.

    === Example ===

    num_experts=4, num_pairs=8, BLOCK_SIZE=4
    topk_ids (flat) = [2, 3, 0, 2, 1, 0, 3, 1]

    Program 0 processes pairs 0-3: expert IDs [2, 3, 0, 2]
      atomic_add(tokens_per_expert[2], 1) → [0, 0, 1, 0]
      atomic_add(tokens_per_expert[3], 1) → [0, 0, 1, 1]
      atomic_add(tokens_per_expert[0], 1) → [1, 0, 1, 1]
      atomic_add(tokens_per_expert[2], 1) → [1, 0, 2, 1]

    Program 1 processes pairs 4-7: expert IDs [1, 0, 3, 1]
      (runs CONCURRENTLY with Program 0 on different SMs!)
      atomic_add(tokens_per_expert[1], 1) → [1, 1, 2, 1]
      atomic_add(tokens_per_expert[0], 1) → [2, 1, 2, 1]
      atomic_add(tokens_per_expert[3], 1) → [2, 1, 2, 2]
      atomic_add(tokens_per_expert[1], 1) → [2, 2, 2, 2]

    Final result: [2, 2, 2, 2] — each expert got 2 pairs. ✓

    === Why atomic_add? ===

    Multiple programs run in parallel on different SMs (streaming multi-
    processors). If two programs try to increment the same expert's counter
    simultaneously without atomics, one increment would be lost (classic
    read-modify-write race condition). `tl.atomic_add` uses hardware-level
    atomic instructions that serialize conflicting writes to the SAME
    address while allowing non-conflicting writes to DIFFERENT addresses
    to proceed in full parallel.
    """
    # Which chunk of pairs does this program handle?
    pid = tl.program_id(axis=0)

    # Compute the pair indices for this program.
    # offs = [pid*BS, pid*BS+1, ..., pid*BS+BS-1]
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Mask: the last program may extend beyond num_pairs.
    mask = offs < num_pairs

    # Load expert IDs for this chunk.
    expert_ids = tl.load(topk_ids_ptr + offs, mask=mask)

    # Atomically increment each expert's counter.
    # Multiple programs may count pairs for the same expert concurrently —
    # the hardware serializes only the conflicting accesses (same address),
    # while non-conflicting ones (different experts) proceed in parallel.
    tl.atomic_add(tokens_per_expert_ptr + expert_ids, 1, mask=mask)


@triton.jit
def _moe_scatter_kernel(
    # --- Pointers ---
    topk_ids_ptr,              # [num_pairs] flattened expert assignments (int32)
    sorted_token_ids_ptr,      # [max_padded] output buffer (pre-filled with sentinel)
    write_counters_ptr,        # [E] atomic write pointers (init to expert_offsets[:E])
    # --- Scalars ---
    num_pairs,                 # Total number of token-expert pairs (T * K)
    # --- Compile-time constants ---
    BLOCK_SIZE: tl.constexpr,  # Pairs processed per Triton program (e.g., 256)
):
    """
    Scatter pair indices into expert-sorted positions using atomic slot
    reservation. Replaces the old argsort + scatter approach with a
    single-pass O(n) atomic scatter.

    === How it works ===

    Before calling this kernel, the caller initializes:
        write_counters[e] = expert_offsets[e]

    This is the starting write position for expert e in sorted_token_ids.

    For each pair, the kernel:
      1. Loads the pair's expert ID
      2. Atomically reserves a slot:
           slot = atomic_add(write_counters[expert], 1)
         atomic_add returns the OLD value = the position to write to,
         then increments the counter (so the next pair for this expert
         gets the next position — like taking a number at a deli counter).
      3. Writes the pair index into sorted_token_ids[slot]

    === Example ===

    4 experts, block_size=4, num_pairs=8
    topk_ids (flat) = [2, 3, 0, 2, 1, 0, 3, 1]

    expert_offsets   = [0, 4, 8, 12, 16]
    write_counters   = [0, 4, 8, 12]  (copied from expert_offsets[:4])
    sorted_token_ids = [8,8,8,8, 8,8,8,8, 8,8,8,8, 8,8,8,8]  (sentinel=8)

    Program 0 processes pairs 0-3 (expert IDs [2, 3, 0, 2]):
      Pair 0 (expert 2): slot=atomic_add(counters[2],1)=8  → ids[8]=0
      Pair 1 (expert 3): slot=atomic_add(counters[3],1)=12 → ids[12]=1
      Pair 2 (expert 0): slot=atomic_add(counters[0],1)=0  → ids[0]=2
      Pair 3 (expert 2): slot=atomic_add(counters[2],1)=9  → ids[9]=3

    Program 1 processes pairs 4-7 (expert IDs [1, 0, 3, 1]):
      Pair 4 (expert 1): slot=atomic_add(counters[1],1)=4  → ids[4]=4
      Pair 5 (expert 0): slot=atomic_add(counters[0],1)=1  → ids[1]=5
      Pair 6 (expert 3): slot=atomic_add(counters[3],1)=13 → ids[13]=6
      Pair 7 (expert 1): slot=atomic_add(counters[1],1)=5  → ids[5]=7

    Final sorted_token_ids (grouped by expert):
      Expert 0: [2, 5, 8, 8]   ← pairs for expert 0, padded with sentinel
      Expert 1: [4, 7, 8, 8]   ← pairs for expert 1, padded with sentinel
      Expert 2: [0, 3, 8, 8]   ← pairs for expert 2, padded with sentinel
      Expert 3: [1, 6, 8, 8]   ← pairs for expert 3, padded with sentinel
    Exactly the sorted order the GEMM kernel needs. ✓

    === Why this is CUDA-graph-safe ===

    - No .item() calls — everything stays on GPU
    - All buffers are pre-allocated to fixed max size
    - The scatter writes DIFFERENT data each step (routing changes), but
      buffer SIZES and kernel GRID are always the same
    - CUDA graphs replay the same kernel launches with the same tensor
      addresses — they don't care about data values, only structure
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < num_pairs

    # Load expert IDs for this chunk.
    expert_ids = tl.load(topk_ids_ptr + offs, mask=mask)

    # Atomically reserve a write slot for each pair.
    # Returns the OLD counter value = the position to write to.
    # After the add, the counter is incremented for the next pair.
    slots = tl.atomic_add(write_counters_ptr + expert_ids, 1, mask=mask)

    # Write pair index into the reserved slot.
    tl.store(sorted_token_ids_ptr + slots, offs.to(tl.int32), mask=mask)


# ---------------------------------------------------------------------------
# Buffer Allocation Helper
# ---------------------------------------------------------------------------

def _allocate_sorting_buffers(
    max_num_tokens: int,
    top_k: int,
    num_experts: int,
    block_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Allocate fixed-size buffers for CUDA-graph-safe MoE token sorting.

    All buffers are sized for the WORST CASE to guarantee that no
    reallocation is ever needed during CUDA graph replay:

        max_pairs  = max_num_tokens * top_k
        max_padded = max_pairs + num_experts * (block_size - 1)
        max_blocks = max_padded // block_size

    The worst-case padding occurs when every expert receives at least one
    pair, requiring each to be padded up to a full block_size. Each expert
    can waste at most (block_size - 1) padding slots.

    Example: T=512, K=8, E=128, block_size=64
        max_pairs  = 512 * 8 = 4096 total token-expert pairs
        max_padded = 4096 + 128 * 63 = 12160 total slots after padding
        max_blocks = 12160 / 64 = 190 blocks for the GEMM kernel

    In practice, actual usage is much smaller (tokens cluster among popular
    experts). Unused slots are filled with sentinel values and the GEMM
    kernel early-exits for blocks beyond the real data.

    Args:
        max_num_tokens: Maximum tokens in any single forward pass.
        top_k: Number of experts per token (e.g., 8 for Qwen3-30B-A3B).
        num_experts: Total number of (global) experts.
        block_size: BLOCK_SIZE_M for the Triton GEMM kernel (e.g., 64).
        device: CUDA device for tensor allocation.

    Returns:
        dict of pre-allocated tensors for moe_align_block_size.
    """
    max_pairs = max_num_tokens * top_k
    max_padded = max_pairs + num_experts * (block_size - 1)
    max_blocks = max_padded // block_size

    return {
        # sorted_token_ids: [max_padded] int32
        # After sorting, pair indices grouped by expert, padded with sentinels.
        # The GEMM kernel masks out entries >= num_valid_tokens.
        'sorted_token_ids': torch.empty(
            max_padded, dtype=torch.int32, device=device,
        ),

        # expert_ids: [max_blocks] int32
        # Maps each block of BLOCK_SIZE_M tokens to its expert.
        # Blocks beyond num_tokens_post_padded have stale/clamped IDs but
        # are early-exited by the kernel before the ID is ever used.
        'expert_ids': torch.empty(
            max_blocks, dtype=torch.int32, device=device,
        ),

        # num_tokens_post_padded: [1] int32
        # Total valid slots after padding (stays on GPU — no .item()!).
        # The GEMM kernel reads this via tl.load() for early-exit.
        'num_tokens_post_padded': torch.empty(
            1, dtype=torch.int32, device=device,
        ),

        # tokens_per_expert: [E] int32
        # Scratch: histogram of pairs per expert. Zeroed each call.
        'tokens_per_expert': torch.empty(
            num_experts, dtype=torch.int32, device=device,
        ),

        # expert_offsets: [E+1] int32
        # Cumulative padded counts. expert_offsets[e] = start position for
        # expert e. expert_offsets[E] = total = num_tokens_post_padded.
        'expert_offsets': torch.empty(
            num_experts + 1, dtype=torch.int32, device=device,
        ),

        # write_counters: [E] int32
        # Scratch for the scatter kernel. Initialized to expert_offsets[:E]
        # before each scatter, then atomically incremented.
        'write_counters': torch.empty(
            num_experts, dtype=torch.int32, device=device,
        ),

        # block_positions: [max_blocks] int32
        # Pre-computed arithmetic sequence: [0, block_size, 2*block_size, ...].
        # Used by searchsorted to map block indices → expert IDs.
        # Computed once and never changes.
        'block_positions': (
            torch.arange(max_blocks, dtype=torch.int32, device=device)
            * block_size
        ),
    }


# ---------------------------------------------------------------------------
# Token Sorting (moe_align_block_size) — CUDA-Graph-Safe Version
# ---------------------------------------------------------------------------
# Before we can run the fused GEMM kernel, we need to reorganize tokens so
# that all tokens assigned to the same expert are contiguous in memory.
#
# Why padding? The Triton kernel processes tokens in blocks of BLOCK_SIZE_M.
# If expert 5 has 7 tokens and BLOCK_SIZE_M=64, we pad to 64 so the kernel
# can process a full tile. Padded slots use token_id = num_valid_tokens,
# which the kernel masks out.
#
# This version writes into PRE-ALLOCATED buffers (no return value) and uses
# Triton kernels + searchsorted instead of .item() / repeat_interleave.
# ---------------------------------------------------------------------------

def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    expert_offsets: torch.Tensor,
    write_counters: torch.Tensor,
    block_positions: torch.Tensor,
    expert_map: torch.Tensor | None = None,
) -> None:
    """
    Sort token-expert pairs by expert, writing into pre-allocated buffers.

    This function is CUDA-graph-safe: no .item() calls, no dynamic-size
    allocations, no data-dependent output sizes. All buffers are pre-allocated
    to worst-case max by _allocate_sorting_buffers().

    Args:
        topk_ids: [num_tokens, top_k] — which experts each token selected.
        block_size: BLOCK_SIZE_M for the Triton kernel (e.g. 64).
        num_experts: Total (global) number of experts.
        sorted_token_ids: [max_padded] pre-allocated output buffer.
        expert_ids: [max_blocks] pre-allocated output buffer.
        num_tokens_post_padded: [1] pre-allocated scalar buffer.
        tokens_per_expert: [E] scratch buffer for histogram.
        expert_offsets: [E+1] scratch buffer for cumulative offsets.
        write_counters: [E] scratch buffer for scatter kernel.
        block_positions: [max_blocks] pre-computed [0, BS, 2*BS, ...].
        expert_map: [E] global→local expert ID mapping (EP), or None.

    Returns nothing — results are written into the pre-allocated buffers.

    Example with 4 tokens, top_k=2, 4 experts, block_size=4:
        topk_ids = [[2,3], [0,2], [1,0], [3,1]]

        Flattened: [2, 3, 0, 2, 1, 0, 3, 1]
        Pair indices:  0  1  2  3  4  5  6  7

        After sorting + padding to block_size=4:
          Expert 0: [2, 5, 8, 8]    (8 = sentinel = num_pairs)
          Expert 1: [4, 7, 8, 8]
          Expert 2: [0, 3, 8, 8]
          Expert 3: [1, 6, 8, 8]

        sorted_token_ids = [2,5,8,8, 4,7,8,8, 0,3,8,8, 1,6,8,8, ...]
                                                         ^^^ rest is sentinel
        expert_ids = [0, 1, 2, 3, ...]
                                   ^^^ rest is clamped/stale (early-exited)
        num_tokens_post_padded = [16]  (on GPU, never pulled to CPU)
    """
    num_tokens = topk_ids.size(0)
    top_k = topk_ids.size(1)
    num_pairs = num_tokens * top_k  # total token-expert pairs

    # Convert to int32 for Triton atomics. Expert IDs are small integers
    # (< num_experts, typically 128), so int32 is more than sufficient.
    # topk_ids is int64 (from torch.topk), but int32 is needed because
    # the Triton atomic operations target int32 buffers.
    flat_ids = topk_ids.flatten().int()

    # === Step 1: Count pairs per expert (Triton histogram kernel) ===
    #
    # Zero the histogram, then count via parallel atomic increments.
    # This replaces torch.bincount which is likely graph-safe but we use
    # the Triton kernel for consistency and educational value.
    #
    # Grid: ceil(num_pairs / 256) programs, each processing 256 pairs.
    # With 4096 pairs (512 tokens × 8 top_k), that's 16 programs.
    tokens_per_expert.zero_()
    HISTOGRAM_BLOCK_SIZE = 256
    histogram_grid = (triton.cdiv(num_pairs, HISTOGRAM_BLOCK_SIZE),)
    _moe_histogram_kernel[histogram_grid](
        flat_ids, tokens_per_expert, num_pairs,
        BLOCK_SIZE=HISTOGRAM_BLOCK_SIZE,
    )

    # === Step 2: Pad each expert's count to block_size alignment ===
    #
    # Element-wise ops on fixed-size GPU tensors → graph-safe.
    # Example: counts [7, 0, 3, 5] with block_size=4 → [8, 0, 4, 8]
    tokens_per_expert_padded = (
        (tokens_per_expert + block_size - 1) // block_size * block_size
    )

    # === Step 3: Compute expert offsets via cumulative sum ===
    #
    # expert_offsets[e] = starting index in sorted_token_ids for expert e.
    # Example: padded [8, 0, 4, 8] → offsets [0, 8, 8, 12, 20]
    # Expert 0 owns positions [0, 8), expert 1 owns [8, 8) (empty),
    # expert 2 owns [8, 12), expert 3 owns [12, 20).
    expert_offsets.zero_() #expert_offsets[0] = 0
    expert_offsets[1:num_experts + 1] = tokens_per_expert_padded.cumsum(0)

    # === Step 4: Total padded count stays as GPU tensor (NO .item()!) ===
    #
    # CRITICAL: This is the key change that makes CUDA graphs possible.
    #
    # Old code (graph-BREAKING):
    #   num_tokens_post_padded = tokens_per_expert_padded.sum().item()
    #   ^^^ .item() forces a CUDA synchronize — the CPU blocks until the
    #   GPU finishes ALL pending work, then copies one int to CPU RAM.
    #   During CUDA graph capture, this is ILLEGAL: the operations haven't
    #   actually executed yet (they're being recorded), so there's no value
    #   to copy. PyTorch raises "CUDA error: illegal memory access".
    #
    # New code (graph-SAFE):
    #   num_tokens_post_padded[0] = expert_offsets[num_experts]
    #   ^^^ GPU-to-GPU copy of a scalar. No CPU involvement. The kernel
    #   reads this value via tl.load(num_tokens_post_padded_ptr) to
    #   decide whether to early-exit for padding blocks.
    num_tokens_post_padded[0] = expert_offsets[num_experts]

    # === Step 5: Fill sorted_token_ids with sentinel value ===
    #
    # Old code (graph-BREAKING):
    #   sorted_token_ids = torch.full((num_tokens_post_padded,), ...)
    #   ^^^ Dynamic size from .item() result → variable allocation
    #
    # New code (graph-SAFE):
    #   sorted_token_ids.fill_(num_pairs)
    #   ^^^ Pre-allocated buffer, fixed size. fill_ is a standard CUDA
    #   kernel on a fixed-size tensor. The sentinel value (num_pairs) is
    #   a Python int computed from tensor .size() which is constant for
    #   a given CUDA graph.
    #
    # The sentinel value (= num_pairs = T*K) is ≥ all valid pair indices
    # (which are 0..num_pairs-1). The GEMM kernel checks:
    #   if offs_token >= num_valid_tokens: skip (mask out)
    # So padding slots produce zero output — correct behavior.
    sorted_token_ids.fill_(num_pairs)

    # === Step 6: Build expert_ids via searchsorted ===
    #
    # expert_ids[b] = which expert owns block b of BLOCK_SIZE_M tokens.
    #
    # Old code (graph-BREAKING):
    #   expert_ids = torch.repeat_interleave(arange, blocks_per_expert)
    #   ^^^ Output size = sum(blocks_per_expert) = num_tokens_post_padded
    #   / block_size, which is DATA-DEPENDENT (varies with routing).
    #   CUDA graphs require fixed tensor sizes.
    #
    # New code (graph-SAFE):
    #   searchsorted on pre-allocated max_blocks-sized tensor.
    #   Output size is always max_blocks (fixed).
    #
    # How searchsorted maps block positions to expert IDs:
    #
    #   expert_offsets   = [0, 64, 192, 256]  (cumulative padded counts)
    #   block_positions  = [0, 64, 128, 192]  (= [0*64, 1*64, 2*64, 3*64])
    #
    #   searchsorted(offsets, 0,   right=True) = 1 → expert 0  ✓
    #   searchsorted(offsets, 64,  right=True) = 2 → expert 1  ✓
    #   searchsorted(offsets, 128, right=True) = 2 → expert 1  ✓
    #   searchsorted(offsets, 192, right=True) = 3 → expert 2  ✓
    #
    #   right=True means: find index i where offsets[i-1] <= pos < offsets[i]
    #   Then expert = i - 1.
    #
    # For blocks beyond the actual data (position >= num_tokens_post_padded),
    # searchsorted returns num_experts+1, clamped to num_experts-1. These
    # blocks are early-exited by the GEMM kernel before the expert ID is
    # ever used, so the clamped value doesn't matter.
    num_blocks = expert_ids.size(0)
    expert_ids[:] = (
        torch.searchsorted(
            expert_offsets.contiguous(),
            block_positions[:num_blocks].contiguous(),
            right=True,
        ) - 1
    )
    expert_ids.clamp_(0, num_experts - 1)

    # === Step 7: Scatter pair indices into sorted positions ===
    #
    # Initialize write_counters to expert_offsets[:E]. Each counter tracks
    # the next write position for that expert. The Triton scatter kernel
    # atomically increments counters to reserve slots, then writes pair
    # indices into sorted_token_ids at the reserved positions.
    #
    # After this step:
    #   sorted_token_ids[offsets[e] : offsets[e]+count[e]] = pair indices
    #   sorted_token_ids[offsets[e]+count[e] : offsets[e+1]] = sentinel
    write_counters.copy_(expert_offsets[:num_experts])
    SCATTER_BLOCK_SIZE = 256
    scatter_grid = (triton.cdiv(num_pairs, SCATTER_BLOCK_SIZE),)
    _moe_scatter_kernel[scatter_grid](
        flat_ids, sorted_token_ids, write_counters, num_pairs,
        BLOCK_SIZE=SCATTER_BLOCK_SIZE,
    )

    # === Step 8: Apply expert_map for Expert Parallelism ===
    #
    # In EP mode, expert_map[global_id] = local_id (or -1 for non-local).
    # Remap so the GEMM kernel indexes into local expert weights.
    # Blocks with expert_id == -1 → kernel writes zeros and returns early.
    if expert_map is not None:
        expert_ids[:] = expert_map[expert_ids.long()]


# ---------------------------------------------------------------------------
# Triton FusedMoE Kernel
# ---------------------------------------------------------------------------
# This kernel computes: C[token, :] = A[token, :] @ B[expert, :, :].T
# with optional multiplication by the router weight.
#
# It is a standard tiled GEMM, but with one twist: the "M" dimension
# (rows of A) is not a contiguous range of tokens. Instead, it's a
# block of sorted_token_ids that all map to the same expert. The kernel
# looks up which expert to use from expert_ids[block_m_index].
#
# Memory layout:
#   A: [num_tokens, K]           — input hidden states (shared across experts)
#   B: [num_experts, N, K]       — stacked expert weights (N=output_dim, K=input_dim)
#   C: [num_tokens * top_k, N]   — output (one row per token-expert pair)
#
# The kernel is launched with a 1D grid of (num_m_blocks * num_n_blocks) blocks.
# Each block computes a [BLOCK_SIZE_M, BLOCK_SIZE_N] tile of the output.
# ---------------------------------------------------------------------------

@triton.jit
def fused_moe_kernel(
    # Pointers to data tensors
    a_ptr,                          # Input activations [num_tokens, K]
    b_ptr,                          # Expert weights [E, N, K]
    c_ptr,                          # Output [num_tokens * top_k, N]
    topk_weights_ptr,               # Router weights [num_tokens * top_k]
    sorted_token_ids_ptr,           # Sorted pair indices from moe_align_block_size
    expert_ids_ptr,                 # Expert ID per M-block
    num_tokens_post_padded_ptr,     # Total padded token count (scalar)
    # Matrix dimensions
    N: tl.constexpr,                # Output dimension (columns of B, rows of B in memory)
    K: tl.constexpr,                # Hidden dimension (shared dim of A and B)
    EM,                             # Total number of sorted token slots (padded)
    num_valid_tokens,               # Actual number of token-expert pairs (unpadded)
    # Strides (elements to skip when moving by 1 along a dimension)
    stride_am,                      # A: stride along token dimension
    stride_ak,                      # A: stride along hidden dimension
    stride_be,                      # B: stride along expert dimension
    stride_bn,                      # B: stride along output dimension
    stride_bk,                      # B: stride along hidden dimension
    stride_cm,                      # C: stride along token dimension
    stride_cn,                      # C: stride along output dimension
    # Compile-time constants
    BLOCK_SIZE_M: tl.constexpr,     # Tile height (tokens per block), e.g. 64
    BLOCK_SIZE_N: tl.constexpr,     # Tile width (output cols per block), e.g. 64
    BLOCK_SIZE_K: tl.constexpr,     # Inner dimension tile, e.g. 32
    GROUP_SIZE_M: tl.constexpr,     # M-blocks grouped together for L2 cache reuse
    MUL_ROUTED_WEIGHT: tl.constexpr,  # Whether to multiply by router weights
    top_k: tl.constexpr,           # Number of experts per token
    compute_type: tl.constexpr,     # Output dtype (e.g. tl.bfloat16)
):
    """
    Fused MoE GEMM kernel: computes expert(input) for all token-expert pairs
    in a single kernel launch.

    The tiling strategy is the standard 2D tiled GEMM from Triton tutorials:

        for each tile (m_block, n_block):
            accumulator = 0
            for k_block in range(K // BLOCK_SIZE_K):
                accumulator += A_tile @ B_tile
            C_tile = accumulator

    The twist is that m_block indexes into sorted_token_ids (not raw token IDs),
    and the expert for this block is looked up from expert_ids[m_block].

    GROUP_SIZE_M groups adjacent M-blocks so they are processed by nearby thread
    blocks. This improves L2 cache hit rate because adjacent M-blocks often share
    the same expert (and thus the same B weights), so the weights stay in L2.
    """
    # --- Block-to-tile mapping with grouped ordering ---
    # The grid is 1D with pid in [0, num_m_blocks * num_n_blocks).
    # We map pid -> (pid_m, pid_n) with "grouped" ordering: GROUP_SIZE_M
    # adjacent M-blocks share N-blocks before moving to the next M-group.
    # This means thread blocks that share the same expert weights (same M-range)
    # are scheduled close together, improving L2 cache reuse for B.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --- Early exit for padding blocks ---
    # Some blocks at the tail are pure padding (no real tokens).
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # --- Load token indices for this M-block ---
    # sorted_token_ids[pid_m * BLOCK_SIZE_M : (pid_m+1) * BLOCK_SIZE_M]
    # These are "pair indices" into the flattened topk_ids. To get the
    # actual token index in A, we divide by top_k (because each token
    # appears top_k times in the sorted list — once per expert it was
    # routed to).
    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs_m)
    offs_token = offs_token.to(tl.int64)

    # Mask: only process real tokens, not padding sentinels.
    # Padding slots have offs_token == num_valid_tokens (or greater).
    token_mask = offs_token < num_valid_tokens

    # --- Look up which expert this block processes ---
    # All tokens in this block belong to the same expert (by construction
    # from the sorting step).
    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Expert Parallelism: if this expert is not on our GPU, write zeros.
    # The expert_map remapped non-local experts to -1.
    if off_expert == -1:
        # Write zeros so the subsequent all-reduce produces correct results
        # (this GPU contributes 0 for non-local experts).
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
        c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
        tl.store(c_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type), mask=c_mask)
        return

    # --- Set up A and B pointers ---
    # A pointers: we index into row (offs_token // top_k) of A.
    # Division by top_k recovers the original token index (since each token
    # appears top_k times in the sorted list).
    # B pointers: we index into expert off_expert's weight matrix.
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        (offs_token[:, None] // top_k) * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + offs_n[None, :] * stride_bn
        + offs_k[:, None] * stride_bk
    )

    # --- Tiled matrix multiplication ---
    # Standard inner loop: accumulate partial products over K dimension.
    # We accumulate in float32 regardless of input dtype for numerical
    # stability (especially important for bf16 where the mantissa is only
    # 8 bits). The final result is cast to compute_type at the end.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load a [BLOCK_SIZE_M, BLOCK_SIZE_K] tile from A
        k_mask = offs_k[None, :] < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & k_mask, other=0.0)

        # Load a [BLOCK_SIZE_K, BLOCK_SIZE_N] tile from B
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # Accumulate: [M, K] @ [K, N] -> [M, N]
        accumulator += tl.dot(a, b)

        # Advance pointers to the next K-tile
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # --- Apply router weights ---
    # Each token-expert pair has a routing weight (from softmax of router
    # logits, after top-k selection). We multiply here to weight each
    # expert's contribution before summing across top-k later.
    # This multiplication MUST happen in float32 before casting to output
    # dtype for numerical stability.
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0.0)
        accumulator *= moe_weight[:, None]

    # Cast to output dtype
    accumulator = accumulator.to(compute_type)

    # --- Write output tile ---
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# ---------------------------------------------------------------------------
# Python Wrapper: invoke the Triton kernel
# ---------------------------------------------------------------------------

def invoke_fused_moe_kernel(
    A: torch.Tensor,           # [num_tokens, K]
    B: torch.Tensor,           # [num_experts, N, K]
    C: torch.Tensor,           # [num_tokens * top_k, N]
    topk_weights: torch.Tensor | None,  # [num_tokens * top_k]
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    mul_routed_weight: bool,
):
    """
    Launch the Triton FusedMoE kernel with fixed block sizes.

    We use conservative block sizes that work well across GPU architectures:
    - BLOCK_SIZE_M=64: good balance between parallelism and register pressure
    - BLOCK_SIZE_N=64: matches typical hidden dims (multiples of 64)
    - BLOCK_SIZE_K=32: small enough to fit in registers, large enough for throughput
    - GROUP_SIZE_M=8: groups 8 M-blocks for L2 cache reuse of expert weights

    vllm uses autotuning to find optimal configs per (E, N, device). We use
    fixed values for simplicity — the performance difference is typically <20%.

    === CUDA Graph Compatibility ===

    EM = sorted_token_ids.size(0). With pre-allocated buffers, this is always
    max_padded (the worst-case size, computed once during warmup). This means
    the grid size is FIXED regardless of the actual routing this step — exactly
    what CUDA graphs require.

    Blocks beyond the actual num_tokens_post_padded early-exit in the kernel:
        num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

    So we launch more blocks than needed, but the excess ones exit immediately
    after reading one int32 from GPU memory — negligible overhead.
    """
    # EM = total sorted slots. With pre-allocated buffers, this is max_padded
    # (constant), giving a fixed grid for CUDA graphs. Without pre-allocated
    # buffers (eager fallback), it matches the actual padded count.
    EM = sorted_token_ids.size(0)
    N = B.size(1)
    K = B.size(2)
    num_valid_tokens = A.size(0) * top_k

    # Grid: one thread block per (m_block, n_block) tile.
    # With pre-allocated buffers: grid is constant (max_padded / 64 * N / 64).
    # Excess blocks early-exit via GPU-resident num_tokens_post_padded.
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Determine compute type from input dtype
    compute_type = tl.bfloat16 if A.dtype == torch.bfloat16 else tl.float16

    fused_moe_kernel[grid](
        A, B, C,
        topk_weights,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        N, K, EM, num_valid_tokens,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_M=64,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
    )


# ---------------------------------------------------------------------------
# High-level fused_moe: orchestrates routing + two GEMM passes + activation
# ---------------------------------------------------------------------------

def fused_moe(
    hidden_states: torch.Tensor,   # [T, hidden_size]
    router_logits: torch.Tensor,   # [T, num_experts]
    w13: torch.Tensor,             # [E, 2 * intermediate_size, hidden_size]
    w2: torch.Tensor,              # [E, hidden_size, intermediate_size]
    top_k: int,
    renormalize: bool = True,
    expert_map: torch.Tensor | None = None,
    sorting_buffers: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """
    Full MoE forward pass: route tokens, compute expert MLPs, combine results.

    The expert MLP is: output = down_proj(SiLU(gate_proj(x)) * up_proj(x))

    We execute this as two fused GEMM passes with an activation in between:

    Pass 1 (w13 = stacked gate_proj + up_proj):
        intermediate = x @ [gate_proj | up_proj].T    →  [T*K, 2*intermediate_size]
        Then apply SiLU(gate) * up (the "SwiGLU" activation)

    Pass 2 (w2 = down_proj, with router weight multiplication):
        output = intermediate @ down_proj.T * router_weight  →  [T*K, hidden_size]

    Finally, sum across the K expert contributions per token:
        final[t] = sum over k of output[t*K + k]

    Args:
        hidden_states: Input from attention layer. [num_tokens, hidden_size]
        router_logits: Raw logits from the router gate. [num_tokens, num_experts]
        w13: Stacked gate+up expert weights. [num_experts, 2*intermediate, hidden]
        w2: Down projection expert weights. [num_experts, hidden, intermediate]
        top_k: Number of experts to activate per token.
        renormalize: Whether to renormalize routing weights to sum to 1.
        expert_map: Global-to-local expert ID mapping for EP. None = no EP.
        sorting_buffers: Pre-allocated sorting buffers from _allocate_sorting_buffers().
            If None, buffers are allocated on-the-fly (eager mode, no CUDA graphs).
            If provided, moe_align_block_size writes into these fixed-size buffers,
            making the entire function CUDA-graph-safe.

    Returns:
        output: [num_tokens, hidden_size]
    """
    num_tokens = hidden_states.size(0)
    hidden_size = hidden_states.size(1)
    num_experts = router_logits.size(1)
    intermediate_size = w2.size(2)  # w2 is [E, hidden, intermediate]

    # --- Step 1: Top-K routing ---
    # Compute softmax over router logits to get expert probabilities,
    # then pick the top-K experts per token.
    routing_weights = torch.softmax(router_logits.float(), dim=-1)
    topk_weights, topk_ids = torch.topk(routing_weights, top_k, dim=-1)

    # Optionally renormalize so the K selected weights sum to 1.
    # This is controlled by norm_topk_prob in the model config.
    # Rationale: after top-k selection, the weights no longer sum to 1.
    # Renormalizing ensures the output scale doesn't depend on how much
    # probability mass was in the top-K vs the tail.
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(hidden_states.dtype)

    # --- Step 2: Sort tokens by expert ---
    # This prepares the inputs for the Triton kernel. See moe_align_block_size
    # docstring for the full explanation.
    #
    # Two paths:
    #   1. sorting_buffers provided → CUDA-graph-safe path. Writes into
    #      pre-allocated fixed-size buffers. No dynamic allocation.
    #   2. sorting_buffers is None → eager path. Allocates temporary buffers.
    #      Used for standalone calls or when CUDA graphs are not needed.
    if sorting_buffers is None:
        # Eager fallback: allocate temporary buffers on-the-fly.
        # This path is NOT CUDA-graph-safe (dynamic allocation), but is
        # convenient for testing or prefill (which always runs eagerly).
        sorting_buffers = _allocate_sorting_buffers(
            num_tokens, top_k, num_experts, MOE_BLOCK_SIZE_M,
            hidden_states.device,
        )

    moe_align_block_size(
        topk_ids, block_size=MOE_BLOCK_SIZE_M, num_experts=num_experts,
        expert_map=expert_map,
        **sorting_buffers,
    )

    # Read sorting results from the buffers.
    sorted_token_ids = sorting_buffers['sorted_token_ids']
    expert_ids = sorting_buffers['expert_ids']
    num_tokens_post_padded = sorting_buffers['num_tokens_post_padded']

    # --- Step 3: Pass 1 — gate+up projection ---
    # w13 shape: [E, 2*intermediate_size, hidden_size]
    # Output shape: [T*K, 2*intermediate_size]
    # We DON'T multiply router weights here — they go on Pass 2.
    intermediate = torch.empty(
        num_tokens * top_k, 2 * intermediate_size,
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    invoke_fused_moe_kernel(
        A=hidden_states,
        B=w13,
        C=intermediate,
        topk_weights=topk_weights.flatten(),
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        top_k=top_k,
        mul_routed_weight=False,  # Don't apply weights yet
    )

    # --- Step 4: SwiGLU activation ---
    # SwiGLU splits the 2*intermediate tensor in half:
    #   gate_output = first half,  up_output = second half
    #   activated = SiLU(gate_output) * up_output
    # This is the same activation used in dense Qwen3 (SiluAndMul).
    gate, up = intermediate.chunk(2, dim=-1)
    intermediate_activated = F.silu(gate) * up

    # --- Step 5: Pass 2 — down projection with router weight multiplication ---
    # w2 shape: [E, hidden_size, intermediate_size]
    # Output shape: [T*K, hidden_size]
    # Router weights are fused into this pass (MUL_ROUTED_WEIGHT=True).
    output = torch.empty(
        num_tokens * top_k, hidden_size,
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
    # CRITICAL: we pass top_k=1 here, NOT the actual top_k.
    #
    # Why? The Triton kernel uses `offs_token // top_k` to convert a pair index
    # (from sorted_token_ids) into a row index for reading A.
    #
    # In Pass 1, A = hidden_states with shape [T, hidden_size]. Each token
    # appears top_k times in sorted_token_ids (once per selected expert).
    # Dividing by top_k recovers the original token index: pair_idx // top_k = token_idx.
    # This is correct — multiple pairs from the same token all read the same row of A.
    #
    # In Pass 2, A = intermediate_activated with shape [T*K, intermediate_size].
    # There's one row per token-expert PAIR, not per token. Each pair needs its
    # own unique row. If we used the real top_k, pair indices from the same token
    # would collapse to the same row (e.g., pairs 0,1 both read row 0), reading
    # the wrong data. Setting top_k=1 makes the kernel index A directly by pair
    # index: pair_idx // 1 = pair_idx, which correctly maps each pair to its row.
    invoke_fused_moe_kernel(
        A=intermediate_activated,
        B=w2,
        C=output,
        topk_weights=topk_weights.flatten(),
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        top_k=1,  # Pass 2: A is per-pair, not per-token. See comment above.
        mul_routed_weight=True,  # Fuse router weight multiplication here
    )

    # --- Step 6: Sum across top-K experts per token ---
    # output is [T*K, hidden_size]. Reshape to [T, K, hidden_size] and sum
    # over the K dimension. This combines the weighted contributions from
    # all K selected experts into a single output per token.
    output = output.view(num_tokens, top_k, hidden_size).sum(dim=1)

    return output


# ---------------------------------------------------------------------------
# FusedMoE nn.Module — stores expert weights and handles loading
# ---------------------------------------------------------------------------

class FusedMoE(nn.Module):
    """
    Fused Mixture of Experts layer with stacked expert weights.

    Instead of storing each expert as a separate nn.Module (128 separate MLPs),
    we stack all expert weights into two large tensors:

        w13: [num_experts, 2 * intermediate_size, hidden_size]
             Stacked gate_proj (w1) and up_proj (w3) for all experts.
             "13" = w1 and w3 concatenated along the output dimension.

        w2:  [num_experts, hidden_size, intermediate_size]
             Stacked down_proj (w2) for all experts.

    This layout enables the Triton kernel to access any expert's weights via
    a single index into the first dimension, without pointer chasing through
    nn.ModuleList.

    For Expert Parallelism, only local experts' weights are stored:
        w13: [local_num_experts, 2 * intermediate_size, hidden_size]
        w2:  [local_num_experts, hidden_size, intermediate_size]

    The expert_map tensor maps global expert IDs to local indices (or -1).
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.renormalize = renormalize

        # Sorting buffers are lazily allocated on first forward() call.
        # They are NOT nn.Parameters or registered buffers — they're scratch
        # space that doesn't need to be saved/loaded with the model.
        # See _ensure_buffers() for details.
        self._sorting_buffers: dict[str, torch.Tensor] | None = None

        # --- Expert Parallelism setup ---
        # We reuse the existing TP (Tensor Parallelism) process group for EP.
        # With tp_size GPUs, each GPU holds num_experts // tp_size experts.
        # This works because in nano-vllm, all GPUs see the same tokens
        # (after attention all-reduce), so each GPU can independently compute
        # its local experts and all-reduce the results.
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # Number of experts on this GPU
        self.local_num_experts = num_experts // self.tp_size
        assert num_experts % self.tp_size == 0, (
            f"num_experts ({num_experts}) must be divisible by tp_size ({self.tp_size})"
        )

        # expert_map: [num_experts] tensor.
        # expert_map[global_id] = local_id if expert is on this GPU, else -1.
        # Example with 8 experts, 2 GPUs:
        #   GPU 0: expert_map = [0, 1, 2, 3, -1, -1, -1, -1]
        #   GPU 1: expert_map = [-1, -1, -1, -1, 0, 1, 2, 3]
        expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
        start = self.tp_rank * self.local_num_experts
        end = start + self.local_num_experts
        expert_map[start:end] = torch.arange(self.local_num_experts, dtype=torch.int32)
        # Register as buffer so it moves to GPU with the model but isn't a parameter.
        self.register_buffer("expert_map", expert_map)

        # --- Expert weight tensors ---
        # Only allocate space for local experts (EP saves memory!).
        # w13: gate_proj and up_proj stacked. Each expert's gate_proj is
        #      [intermediate_size, hidden_size] and up_proj is the same,
        #      so stacked = [2 * intermediate_size, hidden_size].
        # w2:  down_proj. Each expert's down_proj is [hidden_size, intermediate_size].
        self.w13 = nn.Parameter(
            torch.empty(self.local_num_experts, 2 * intermediate_size, hidden_size)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.local_num_experts, hidden_size, intermediate_size)
        )

        # Attach custom weight loaders so the generic load_model() function
        # knows how to place HuggingFace per-expert weights into our stacked
        # tensors. See weight_loader_w13 / weight_loader_w2 below.
        self.w13.weight_loader = self.weight_loader_w13
        self.w2.weight_loader = self.weight_loader_w2

    def weight_loader_w13(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ) -> None:
        """
        Load a single expert's gate_proj or up_proj weight into the stacked w13 tensor.

        HuggingFace stores weights as:
            model.layers.L.mlp.experts.E.gate_proj.weight  [intermediate, hidden]
            model.layers.L.mlp.experts.E.up_proj.weight    [intermediate, hidden]

        We stack them into w13[local_expert_id]:
            w13[local_id, 0:intermediate, :]                = gate_proj (shard_id="w1")
            w13[local_id, intermediate:2*intermediate, :]   = up_proj   (shard_id="w3")

        The shard_id naming (w1, w3) follows vllm convention where:
            w1 = gate_proj, w2 = down_proj, w3 = up_proj
        """
        # Map global expert ID to local ID via expert_map
        local_id = self.expert_map[expert_id].item()
        if local_id == -1:
            return  # This expert is not on our GPU — skip

        expert_data = param.data[local_id]
        if shard_id == "w1":  # gate_proj → first half
            expert_data[:self.intermediate_size, :].copy_(loaded_weight)
        elif shard_id == "w3":  # up_proj → second half
            expert_data[self.intermediate_size:, :].copy_(loaded_weight)
        else:
            raise ValueError(f"Unknown shard_id for w13: {shard_id}")

    def weight_loader_w2(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
        shard_id: str,
    ) -> None:
        """
        Load a single expert's down_proj weight into the stacked w2 tensor.

        HuggingFace: model.layers.L.mlp.experts.E.down_proj.weight [hidden, intermediate]
        We store:    w2[local_expert_id, :, :] = down_proj

        shard_id is always "w2" for down_proj.
        """
        local_id = self.expert_map[expert_id].item()
        if local_id == -1:
            return  # Not our expert

        assert shard_id == "w2", f"Expected shard_id='w2', got '{shard_id}'"
        param.data[local_id].copy_(loaded_weight)

    def _ensure_buffers(self, num_tokens: int, device: torch.device) -> None:
        """
        Lazily allocate fixed-size sorting buffers for CUDA-graph-safe MoE.

        Called on the first forward() pass (typically during model warmup,
        before CUDA graph capture). Once allocated, buffers persist for the
        lifetime of the model and are reused across all subsequent calls.

        === Why lazy allocation? ===

        At __init__ time, model parameters are on CPU — we don't know the
        CUDA device yet, and we don't know max_num_tokens until the engine
        calls forward(). Lazy allocation naturally sizes buffers for the
        actual maximum.

        === Sizing guarantee ===

        The engine calls warmup_model() first with max_num_batched_tokens
        tokens (the largest batch the model will ever see). This triggers
        _ensure_buffers() with the maximum, so all subsequent calls (CUDA
        graph capture with smaller decode batches, and inference) reuse the
        same oversized buffers. No reallocation ever happens during graph
        replay.

        === Buffer lifecycle ===

        1. Model init (__init__):  _sorting_buffers = None
        2. Warmup forward:         _ensure_buffers(max_tokens) → allocate
        3. Graph capture forward:  _ensure_buffers(bs) → no-op (big enough)
        4. Inference replay:       Sorting kernels write into same buffers

        The buffers have FIXED addresses from step 2 onward, which is what
        CUDA graphs require.

        Args:
            num_tokens: Number of tokens in the current forward pass.
            device: CUDA device for tensor allocation.
        """
        if self._sorting_buffers is not None:
            # Already allocated. Verify big enough (should always be True
            # since warmup processes the largest batch, but defensive check).
            existing_capacity = self._sorting_buffers['sorted_token_ids'].size(0)
            needed_capacity = (
                num_tokens * self.top_k
                + self.num_experts * (MOE_BLOCK_SIZE_M - 1)
            )
            if existing_capacity >= needed_capacity:
                return
            # Rare: existing buffers too small. Reallocate.
            # WARNING: This must NOT happen during CUDA graph replay
            # (would change tensor addresses → graph corruption).

        self._sorting_buffers = _allocate_sorting_buffers(
            max_num_tokens=num_tokens,
            top_k=self.top_k,
            num_experts=self.num_experts,
            block_size=MOE_BLOCK_SIZE_M,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: route tokens to experts and compute weighted expert outputs.

        Args:
            hidden_states: [num_tokens, hidden_size] from attention layer.
            router_logits: [num_tokens, num_experts] raw router gate output.

        Returns:
            output: [num_tokens, hidden_size] — weighted sum of expert outputs.
        """
        # Ensure sorting buffers are allocated (no-op after first call).
        # On the very first call (during warmup), this allocates buffers
        # sized for the current (maximum) num_tokens. All subsequent calls
        # reuse these same buffers — fixed addresses, fixed sizes.
        self._ensure_buffers(hidden_states.size(0), hidden_states.device)

        # Run the fused MoE computation with pre-allocated sorting buffers.
        # This makes the entire forward pass CUDA-graph-safe:
        # - moe_align_block_size writes into fixed-size buffers (no .item())
        # - The Triton GEMM kernel grid is fixed (max_padded / BLOCK_SIZE_M)
        # - Excess blocks early-exit via GPU-resident num_tokens_post_padded
        output = fused_moe(
            hidden_states=hidden_states,
            router_logits=router_logits,
            w13=self.w13,
            w2=self.w2,
            top_k=self.top_k,
            renormalize=self.renormalize,
            expert_map=self.expert_map if self.tp_size > 1 else None,
            sorting_buffers=self._sorting_buffers,
        )

        # All-reduce across GPUs to sum partial expert contributions.
        # Each GPU computed output for its local experts only (the rest are zeros).
        # Summing gives the correct result: sum of all experts' weighted outputs.
        # This is the same all-reduce pattern used by RowParallelLinear in the
        # attention layer — EP piggybacks on the existing TP communication group.
        if self.tp_size > 1:
            dist.all_reduce(output)

        return output
