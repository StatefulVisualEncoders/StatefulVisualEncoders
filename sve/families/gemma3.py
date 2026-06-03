"""SVE injection for Gemma-3 (``model_type="gemma3"``).

Vision encoder is SigLIP at ``model.model.vision_tower.vision_model``; blocks at
``.encoder.layers`` are ``SiglipEncoderLayer`` with separate q/k/v projections,
LayerNorms, a 2-Linear GELU MLP, and **no RoPE** (learned absolute positions — so
the SVE block uses ``no_pos_embed=True``). Tensors are dense ``[B,S,D]``.

Note: ``Gemma3MultiModalProjector`` average-pools the 256 SigLIP patch tokens
before the LM, so per-patch cross-image signal is diluted unless spread out — a known
caveat for SVE on Gemma-3 (see paper §3.5).
"""

import types

from ..utils import (
    count_sve_params, ddp_touch, freeze_except_sve, host_num_heads,
    init_block_from_self_attn, make_sve_block, persist_sve_config,
    register_dense_pre_hook, register_seqlens_routing, sve_dense_pre,
)


def _resolve_vision_tower(model):
    for path in ("model.vision_tower.vision_model", "vision_tower.vision_model"):
        mod = model
        for part in path.split("."):
            mod = getattr(mod, part, None)
            if mod is None:
                break
        if mod is not None and hasattr(mod, "encoder") and hasattr(mod.encoder, "layers"):
            return mod
    raise AttributeError("Could not locate SigLIP vision transformer (.encoder.layers).")


def _layer_forward(self, hidden_states, attention_mask=None, **kwargs):
    hidden_states = sve_dense_pre(self, hidden_states)
    residual = hidden_states
    hidden_states, _ = self.self_attn(
        hidden_states=self.layer_norm1(hidden_states), attention_mask=attention_mask, **kwargs,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.mlp(self.layer_norm2(hidden_states))
    hidden_states = residual + hidden_states
    return hidden_states + ddp_touch(self)


def inject(model, *, proj_init_std: float) -> int:
    vision_model = _resolve_vision_tower(model)
    blocks = vision_model.encoder.layers
    cfg = vision_model.config
    hidden_size = cfg.hidden_size
    num_heads = host_num_heads(cfg)
    activation = getattr(cfg, "hidden_act", "gelu_pytorch_tanh") or "gelu_pytorch_tanh"
    p = next(vision_model.parameters())

    for block in blocks:  # inject every ViT block (paper recipe)
        sve_blk = make_sve_block(hidden_size, num_heads, cfg.intermediate_size,
                       proj_init_std=proj_init_std, no_pos_embed=True,
                       dtype=p.dtype, device=p.device,
                       ffn_kind="gelu_fc1_fc2", activation=activation, norm_kind="layer_norm")
        init_block_from_self_attn(sve_blk, block, source_layout="siglip")
        block.sve_block = sve_blk
        block._sve_ctx = None
        block.forward = types.MethodType(_layer_forward, block)

    register_seqlens_routing(model, vision_model)
    register_dense_pre_hook(vision_model, blocks)
    freeze_except_sve(blocks)
    persist_sve_config(model)
    return count_sve_params(blocks)
