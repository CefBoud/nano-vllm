"""
Qwen3 Mixture-of-Experts model for nano-vllm.

=== Architecture Overview ===

Qwen3 MoE is architecturally identical to dense Qwen3 except for one change:
the dense MLP in each decoder layer is replaced by a Sparse MoE block.

Dense Qwen3 decoder layer:
    LayerNorm → Attention → LayerNorm → MLP → residual add

Qwen3 MoE decoder layer:
    LayerNorm → Attention → LayerNorm → SparseMoEBlock → residual add

The SparseMoEBlock contains:
    1. A router gate: small linear [hidden_size → num_experts] that scores each
       expert for each token.
    2. Top-K selection: pick the K highest-scoring experts per token.
    3. FusedMoE: execute the K selected expert MLPs in parallel via the Triton
       kernel, weight by router scores, and sum.

For Qwen3-30B-A3B specifically:
    - 128 experts, 8 selected per token (top_k=8)
    - Each expert MLP has intermediate_size=768 (small per expert)
    - decoder_sparse_step=1 → every layer is MoE (no dense-only layers)
    - No shared experts (unlike Qwen2-MoE which had them)

=== What we reuse from qwen3.py ===

The attention mechanism is identical between dense and MoE Qwen3, so we
import and reuse Qwen3Attention directly. Only the MLP/FFN layer differs.

=== Config differences (dense Qwen3 vs Qwen3 MoE) ===

MoE configs have these extra fields (from HuggingFace config.json):
    - model_type: "qwen3_moe" (instead of "qwen3")
    - num_experts: 128
    - num_experts_per_tok: 8
    - moe_intermediate_size: 768 (per-expert hidden dim)
    - decoder_sparse_step: 1 (every Nth layer is MoE; 1 = all layers)
    - mlp_only_layers: [] (list of layer indices that stay dense)
    - norm_topk_prob: true (renormalize top-k routing weights)
    - intermediate_size: 6144 (used only for dense-only layers, if any)
"""

import torch
from torch import nn
import torch.distributed as dist

# Reuse the attention implementation from dense Qwen3 — it's identical.
# The MoE change is ONLY in the MLP/FFN portion of the decoder layer.
from nanovllm.models.qwen3 import Qwen3Attention, Qwen3MLP

from nanovllm.layers.fused_moe import FusedMoE
from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3MoeSparseMoeBlock(nn.Module):
    """
    Sparse Mixture-of-Experts block that replaces the dense MLP.

    Architecture:
        1. Router gate: linear projection [hidden_size → num_experts]
           - Uses ReplicatedLinear (same weights on every GPU, NOT sharded)
           - This ensures all GPUs make the SAME routing decisions, which is
             critical for EP correctness: if GPU 0 thinks token T goes to
             expert 5 but GPU 1 disagrees, the results after all-reduce will
             be wrong.

        2. FusedMoE: Triton kernel that executes the top-K expert MLPs
           - Handles expert parallelism internally (skips non-local experts)
           - Handles the all-reduce across GPUs

    Why is the router ReplicatedLinear and not sharded?
        The router weight is tiny: [128, 2048] = 256K params = 512KB in bf16.
        Replicating it across GPUs is negligible memory cost, but it guarantees
        all GPUs compute identical routing decisions. This is essential because
        each GPU only computes its local experts — if routing disagreed, the
        all-reduce would sum mismatched partial results.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        moe_intermediate_size: int,
        norm_topk_prob: bool = True,
    ) -> None:
        super().__init__()

        # Router: scores each expert for each token. Output is [T, num_experts].
        # No bias — standard for MoE routers. The softmax in fused_moe() converts
        # raw logits to probabilities before top-k selection.
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

        # FusedMoE: stores all expert weights and handles the fused computation.
        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=num_experts_per_tok,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            renormalize=norm_topk_prob,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Router: compute expert scores for each token.
        # gate output shape: [num_tokens, num_experts]
        router_logits = self.gate(hidden_states)

        # FusedMoE: route tokens to experts, compute expert MLPs, combine results.
        # This handles top-k selection, kernel dispatch, and all-reduce internally.
        output = self.experts(hidden_states, router_logits)

        return output


class Qwen3MoeDecoderLayer(nn.Module):
    """
    Single decoder layer for Qwen3 MoE.

    Identical to Qwen3DecoderLayer (dense) except:
    - The MLP is replaced by a SparseMoEBlock (for MoE layers)
    - Dense layers are still possible via decoder_sparse_step / mlp_only_layers

    The config fields decoder_sparse_step and mlp_only_layers control which
    layers are MoE vs dense:
        - decoder_sparse_step=1: every layer is MoE
        - decoder_sparse_step=2: every other layer is MoE (odd layers)
        - mlp_only_layers=[0,1]: layers 0 and 1 are always dense

    For Qwen3-30B-A3B, decoder_sparse_step=1 and mlp_only_layers=[], so
    ALL layers are MoE. But we implement the general case for correctness.
    """

    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()

        # Attention is identical to dense Qwen3 — reuse directly.
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )

        # Decide whether this layer uses MoE or dense MLP.
        # The decision logic mirrors vllm's qwen3_moe.py:394-407.
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        num_experts = getattr(config, "num_experts", 0)

        is_moe_layer = (
            layer_idx not in mlp_only_layers
            and num_experts > 0
            and (layer_idx + 1) % decoder_sparse_step == 0
        )

        if is_moe_layer:
            # MoE layer: use SparseMoEBlock with expert routing.
            # moe_intermediate_size is the per-expert intermediate dim (e.g. 768),
            # which is much smaller than the dense intermediate_size (e.g. 6144).
            self.mlp = Qwen3MoeSparseMoeBlock(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                moe_intermediate_size=config.moe_intermediate_size,
                norm_topk_prob=getattr(config, "norm_topk_prob", True),
            )
        else:
            # Dense layer: standard MLP (same as dense Qwen3).
            # Uses the full intermediate_size (not moe_intermediate_size).
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass follows the standard pre-norm transformer pattern:
            1. LayerNorm + Attention (with residual)
            2. LayerNorm + MLP/MoE (with residual)

        The residual connection uses a fused RMSNorm+residual-add pattern
        (inherited from nano-vllm's RMSNorm implementation) to reduce memory
        bandwidth: instead of separate norm and add operations, they are
        combined into one pass over the tensor.
        """
        # --- Self-attention with residual ---
        if residual is None:
            # First layer: no prior residual
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            # Subsequent layers: fused residual add + layernorm
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)

        # --- MLP/MoE with residual ---
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class Qwen3MoeModel(nn.Module):
    """
    Qwen3 MoE transformer body (without the LM head).

    Embedding → [DecoderLayer × num_hidden_layers] → Final LayerNorm

    Each decoder layer may be MoE or dense depending on config.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        # Pass layer_idx to each layer so it can decide MoE vs dense.
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    """
    Qwen3 MoE causal language model (full model with LM head).

    === Weight Loading ===

    packed_modules_mapping tells the weight loader how to map HuggingFace
    checkpoint weight names to our fused parameter names:

    Attention weights (same as dense Qwen3):
        q_proj → qkv_proj (shard_id="q")
        k_proj → qkv_proj (shard_id="k")
        v_proj → qkv_proj (shard_id="v")

    Dense MLP weights (only for mlp_only_layers, if any):
        gate_proj → gate_up_proj (shard_id=0)
        up_proj   → gate_up_proj (shard_id=1)

    Expert weights are handled SEPARATELY by the loader (not via packed_modules_mapping)
    because they have a different structure:
        experts.{N}.gate_proj → FusedMoE.w13 (expert_id=N, shard_id="w1")
        experts.{N}.up_proj   → FusedMoE.w13 (expert_id=N, shard_id="w3")
        experts.{N}.down_proj → FusedMoE.w2  (expert_id=N, shard_id="w2")

    The router gate weight (gate.weight) is loaded directly as a ReplicatedLinear
    (no remapping needed).
    """

    # Mapping for attention and dense-MLP weight packing.
    # Expert weights bypass this — they use the expert_params_mapping in loader.py.
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # expert_params_mapping tells the loader how to route per-expert weights
    # from HuggingFace checkpoints into our stacked FusedMoE tensors.
    # Each tuple is: (our_param_suffix, hf_weight_suffix, shard_id)
    # The loader will iterate over all expert_ids and try each mapping.
    expert_params_mapping = [
        ("experts.w13", "gate_proj", "w1"),   # gate_proj → w13, first half
        ("experts.w13", "up_proj", "w3"),     # up_proj   → w13, second half
        ("experts.w2", "down_proj", "w2"),    # down_proj → w2
    ]

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
