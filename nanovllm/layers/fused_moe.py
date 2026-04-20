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
"""

import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Token Sorting (moe_align_block_size)
# ---------------------------------------------------------------------------
# Before we can run the fused kernel, we need to reorganize tokens so that
# all tokens assigned to the same expert are contiguous in memory. This is
# the "preparation" step that feeds the Triton kernel.
#
# Why padding? The Triton kernel processes tokens in blocks of BLOCK_SIZE_M.
# If expert 5 has 7 tokens and BLOCK_SIZE_M=64, we pad to 64 so the kernel
# can process a full tile. Padded slots use token_id = num_valid_tokens,
# which the kernel masks out.
# ---------------------------------------------------------------------------

def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sort token-expert pairs by expert and pad to block_size alignment.

    Args:
        topk_ids: [num_tokens, top_k] — which experts each token selected.
        block_size: BLOCK_SIZE_M for the Triton kernel (e.g. 64).
        num_experts: Total (global) number of experts.
        expert_map: [num_experts] mapping global expert ID -> local ID (or -1).
                    Used for Expert Parallelism to mark non-local experts.

    Returns:
        sorted_token_ids: Flat tensor of token-expert pair indices, sorted by
                          expert, padded so each expert's count is a multiple
                          of block_size. "Token-expert pair index" means the
                          index into the flattened topk_ids — so if token 3
                          selected experts [7, 12], index 6 = 3*2+0 maps to
                          expert 7, and index 7 = 3*2+1 maps to expert 12.
        expert_ids: [num_blocks] — which expert each block of BLOCK_SIZE_M
                    tokens belongs to. -1 for non-local experts (EP).
        num_tokens_post_padded: Scalar tensor — total length of sorted_token_ids
                                after padding.

    Example with 4 tokens, top_k=2, 4 experts, block_size=4:
        topk_ids = [[2,3], [0,2], [1,0], [3,1]]

        Flattened: [2, 3, 0, 2, 1, 0, 3, 1]
        Pair indices:  0  1  2  3  4  5  6  7

        Group by expert:
          Expert 0: pair indices [2, 5]   (from tokens 1 and 2)
          Expert 1: pair indices [4, 7]   (from tokens 2 and 3)
          Expert 2: pair indices [0, 3]   (from tokens 0 and 1)
          Expert 3: pair indices [1, 6]   (from tokens 0 and 3)

        After padding to block_size=4:
          Expert 0: [2, 5, PAD, PAD]    Expert 1: [4, 7, PAD, PAD]
          Expert 2: [0, 3, PAD, PAD]    Expert 3: [1, 6, PAD, PAD]

        sorted_token_ids = [2,5,8,8, 4,7,8,8, 0,3,8,8, 1,6,8,8]
                                                      (8 = num_valid_tokens = padding sentinel)
        expert_ids = [0, 1, 2, 3]  (one per block of 4)
    """
    num_tokens = topk_ids.size(0)
    top_k = topk_ids.size(1)
    num_valid_tokens = num_tokens * top_k  # total token-expert pairs

    # flatten: [T, K] -> [T*K], giving us a flat list of expert assignments
    flat_ids = topk_ids.flatten()

    # Count how many token-expert pairs each expert received.
    # bincount gives us a histogram: tokens_per_expert[e] = number of pairs
    # assigned to expert e.
    tokens_per_expert = torch.bincount(flat_ids, minlength=num_experts)

    # Pad each expert's count up to the next multiple of block_size.
    # This ensures the Triton kernel can process full tiles for every expert.
    tokens_per_expert_padded = (
        (tokens_per_expert + block_size - 1) // block_size * block_size
    )
    num_tokens_post_padded = tokens_per_expert_padded.sum().item()

    # Compute the starting offset for each expert in the sorted output.
    # expert_offsets[e] = sum of padded counts for experts 0..e-1.
    expert_offsets = torch.zeros(
        num_experts + 1, dtype=torch.int32, device=topk_ids.device
    )
    expert_offsets[1:] = tokens_per_expert_padded.cumsum(0)

    # Allocate output arrays.
    # sorted_token_ids: filled with num_valid_tokens (the padding sentinel).
    # Any slot that isn't overwritten stays as the sentinel, which the kernel
    # masks out (token_id >= num_valid_tokens → skip).
    sorted_token_ids = torch.full(
        (num_tokens_post_padded,),
        fill_value=num_valid_tokens,
        dtype=torch.int32,
        device=topk_ids.device,
    )

    # expert_ids: one entry per block, tells the kernel which expert's weights
    # to use for that block of BLOCK_SIZE_M tokens.
    # We use repeat_interleave to expand each expert_idx by its number of blocks.
    # E.g., if expert 0 gets 2 blocks and expert 1 gets 1 block:
    #   expert_ids = [0, 0, 1]
    # This avoids a Python loop with .item() calls.
    blocks_per_expert = tokens_per_expert_padded // block_size
    expert_ids = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=topk_ids.device),
        blocks_per_expert.int(), # this second arg indicates how much each elem from the first arg is repeated
    )

    # Fill sorted_token_ids by scattering each pair index into the right
    # position within each expert's allocated slot range.
    #
    # Strategy: sort pair indices by expert, then compute each pair's position
    # within its expert group using a cumulative count. All done on GPU with
    # no Python loops or .item() calls.
    #
    # Step 1: argsort gives us pair indices ordered by expert assignment.
    # stable=True ensures pairs within the same expert keep their original order.
    order = flat_ids.argsort(stable=True)

    # Step 2: Compute within-expert offsets.
    # After sorting, pairs for expert 0 come first, then expert 1, etc.
    # We need to know each pair's position within its expert's group:
    #   pair 0 of expert 0 → offset 0, pair 1 of expert 0 → offset 1, ...
    # We do this by subtracting the cumulative count at the start of each expert.
    #
    # tokens_per_expert_cumsum[e] = total pairs for experts 0..e-1
    # For each sorted pair, its expert's cumsum gives the starting count,
    # and its position in the sorted array minus that cumsum gives the offset.
    tokens_per_expert_cumsum = torch.zeros(
        num_experts, dtype=torch.int32, device=topk_ids.device
    )
    tokens_per_expert_cumsum[1:] = tokens_per_expert[:-1].cumsum(0)

    # For each pair in sorted order, look up its expert and that expert's
    # cumulative count. Subtracting gives the within-expert index.
    sorted_experts = flat_ids[order]  # expert IDs in sorted order
    cumsum_for_each = tokens_per_expert_cumsum[sorted_experts.long()]
    within_expert_idx = torch.arange(
        num_valid_tokens, dtype=torch.int32, device=topk_ids.device
    ) - cumsum_for_each

    # Step 3: Compute the final write position for each pair.
    # write_pos = expert_offsets[expert] + within_expert_idx
    # expert_offsets already accounts for padding (each expert's block is
    # padded to a multiple of block_size).
    write_positions = expert_offsets[sorted_experts.long()] + within_expert_idx

    # Step 4: Scatter pair indices into sorted_token_ids at the computed positions.
    sorted_token_ids[write_positions.long()] = order.to(torch.int32)

    # Apply expert_map for Expert Parallelism: remap global expert IDs to
    # local IDs. Non-local experts become -1, telling the kernel to skip them.
    if expert_map is not None:
        expert_ids = expert_map[expert_ids.long()]

    num_tokens_post_padded_tensor = torch.tensor(
        [num_tokens_post_padded], dtype=torch.int32, device=topk_ids.device
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded_tensor


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
    """
    EM = sorted_token_ids.size(0)
    N = B.size(1)
    K = B.size(2)
    num_valid_tokens = A.size(0) * top_k

    # Grid: one thread block per (m_block, n_block) tile
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

    Returns:
        output: [num_tokens, hidden_size]
    """
    num_tokens = hidden_states.size(0)
    hidden_size = hidden_states.size(1)
    num_experts = router_logits.size(1)  #w13.size(0)
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
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size=64, num_experts=num_experts, expert_map=expert_map,
    )

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
        # Run the fused MoE computation. expert_map is passed so the kernel
        # skips non-local experts (writes zeros for them).
        output = fused_moe(
            hidden_states=hidden_states,
            router_logits=router_logits,
            w13=self.w13,
            w2=self.w2,
            top_k=self.top_k,
            renormalize=self.renormalize,
            expert_map=self.expert_map if self.tp_size > 1 else None,
        )

        # All-reduce across GPUs to sum partial expert contributions.
        # Each GPU computed output for its local experts only (the rest are zeros).
        # Summing gives the correct result: sum of all experts' weighted outputs.
        # This is the same all-reduce pattern used by RowParallelLinear in the
        # attention layer — EP piggybacks on the existing TP communication group.
        if self.tp_size > 1:
            dist.all_reduce(output)

        return output
