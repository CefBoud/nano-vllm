"""
Weight loader for nano-vllm models from HuggingFace safetensors checkpoints.

=== How weight loading works ===

HuggingFace stores model weights with names like:
    model.layers.0.self_attn.q_proj.weight
    model.layers.0.mlp.gate_proj.weight
    model.layers.0.mlp.experts.5.gate_proj.weight   (MoE only)
    model.layers.0.mlp.gate.weight                   (MoE router)

nano-vllm packs some weights for efficiency:
    q_proj + k_proj + v_proj  → single qkv_proj
    gate_proj + up_proj       → single gate_up_proj

The packed_modules_mapping on each model class tells us how to remap:
    {"q_proj": ("qkv_proj", "q"), "k_proj": ("qkv_proj", "k"), ...}

For MoE models, expert weights need special handling because:
1. They have an expert index embedded in the name: "experts.{N}.gate_proj"
2. They map to stacked tensors (FusedMoE.w13 and FusedMoE.w2) rather than
   individual nn.Linear modules
3. The weight_loader on these parameters expects (param, weight, expert_id, shard_id)

The expert_params_mapping on MoE model classes tells us how to remap:
    [("experts.w13", "gate_proj", "w1"),    # gate_proj → w13, shard_id="w1"
     ("experts.w13", "up_proj",   "w3"),    # up_proj   → w13, shard_id="w3"
     ("experts.w2",  "down_proj", "w2")]    # down_proj → w2,  shard_id="w2"
"""

import os
import re
from glob import glob

import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    """Fallback loader: just copy the weight directly."""
    param.data.copy_(loaded_weight)


# Regex to match expert weight names and extract the expert index.
# Matches patterns like: "model.layers.5.mlp.experts.42.gate_proj.weight"
# Captures: expert_id=42, proj_name="gate_proj"
_EXPERT_WEIGHT_PATTERN = re.compile(
    r"(.+\.mlp\.)experts\.(\d+)\.(gate_proj|up_proj|down_proj)(\.weight)"
)


def load_model(model: nn.Module, path: str):
    """
    Load model weights from HuggingFace safetensors files.

    Handles three categories of weights:

    1. Packed weights (attention QKV, dense MLP gate/up):
       Detected via packed_modules_mapping on the model class.
       Remaps HF names (e.g. "q_proj") to packed names (e.g. "qkv_proj")
       and calls the parameter's weight_loader with a shard_id.

    2. Expert weights (MoE expert gate/up/down projections):
       Detected via regex matching "experts.{N}.{proj}" in the weight name.
       Uses expert_params_mapping to remap to stacked tensor names and calls
       the FusedMoE parameter's weight_loader with (expert_id, shard_id).

    3. Regular weights (everything else):
       Loaded directly via default_weight_loader or the parameter's own loader.
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    expert_params_mapping = getattr(model, "expert_params_mapping", [])

    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():

                # --- Category 2: Expert weights ---
                # Check this FIRST because expert weights also contain
                # "gate_proj" / "up_proj" which would match packed_modules_mapping.
                # We need to intercept them before the generic packed handling.
                expert_match = _EXPERT_WEIGHT_PATTERN.match(weight_name)
                if expert_match and expert_params_mapping:
                    # Parse the expert ID and projection name from the weight name.
                    # Example: "model.layers.5.mlp.experts.42.gate_proj.weight"
                    #   prefix = "model.layers.5.mlp."
                    #   expert_id = 42
                    #   proj_name = "gate_proj"
                    #   suffix = ".weight"
                    prefix = expert_match.group(1)     # "model.layers.5.mlp."
                    expert_id = int(expert_match.group(2))  # 42
                    proj_name = expert_match.group(3)  # "gate_proj"
                    suffix = expert_match.group(4)     # ".weight"

                    # Look up the mapping for this projection type.
                    # expert_params_mapping entries are:
                    #   (our_param_name, hf_proj_name, shard_id)
                    # e.g. ("experts.w13", "gate_proj", "w1")
                    for param_suffix, hf_proj, shard_id in expert_params_mapping:
                        if proj_name == hf_proj:
                            # Build our parameter name:
                            #   "model.layers.5.mlp." + "experts.w13"
                            # NOTE: we do NOT append the ".weight" suffix here because
                            # w13 and w2 are registered as bare nn.Parameter on FusedMoE
                            # (not inside nn.Linear), so their registered name is just
                            # "experts.w13", not "experts.w13.weight".
                            param_name = prefix + param_suffix
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(
                                param,
                                f.get_tensor(weight_name),
                                expert_id=expert_id,
                                shard_id=shard_id,
                            )
                            break
                    continue

                # --- Category 1: Packed weights (QKV, gate/up) ---
                # Check if this weight name matches any packed mapping.
                # Skip expert weights (they were handled above, but this is
                # a safety check in case the regex didn't match).
                packed = False
                for k in packed_modules_mapping:
                    if k in weight_name and "experts" not in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        packed = True
                        break

                if packed:
                    continue

                # --- Category 3: Regular weights ---
                # Direct load: embedding, layernorm, lm_head, router gate, etc.
                param = model.get_parameter(weight_name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, f.get_tensor(weight_name))
