"""SVE injection for InternVL 3.5 (``model_type="internvl"``).

Vision encoder ``InternVLVisionModel`` at ``model.model.vision_tower``; blocks at
``.encoder.layer`` are ``InternVLVisionLayer`` with **separate** q/k/v projections,
optional QK-norm, LayerScale (``lambda_1``/``lambda_2``), a 2-Linear GELU MLP, and
**no RoPE** (absolute positions added at patch-embed — so the SVE block also
uses ``no_pos_embed=True``). Tensors are dense ``[B=n_images, S, D]``; we loop
over images rather than using flash-varlen.
"""

import types

from ..utils import (
    count_sve_params, ddp_touch, freeze_except_sve, host_num_heads,
    init_block_from_self_attn, make_sve_block, persist_sve_config,
    register_dense_pre_hook, register_seqlens_routing, sve_dense_pre,
)


def _resolve_vision_tower(model):
    for path in ("model.vision_tower", "vision_tower"):
        mod = model
        for part in path.split("."):
            mod = getattr(mod, part, None)
            if mod is None:
                break
        if mod is not None and hasattr(mod, "encoder") and hasattr(mod.encoder, "layer"):
            return mod
    raise AttributeError("Could not locate InternVL vision_tower (.encoder.layer).")


def _norm_kind(norm_type) -> str:
    return "rms_norm" if norm_type is not None and "rms" in str(norm_type).lower() else "layer_norm"


def _layer_forward(self, hidden_states, **kwargs):
    hidden_states = sve_dense_pre(self, hidden_states)
    attn_out, _ = self.attention(self.layernorm_before(hidden_states))
    hidden_states = self.lambda_1 * attn_out + hidden_states
    layer_out = self.dropout(self.mlp(self.layernorm_after(hidden_states)))
    if self.lambda_2 is not None:
        layer_out = self.lambda_2 * layer_out
    return layer_out + hidden_states + ddp_touch(self)


def inject(model, *, proj_init_std: float) -> int:
    vision_model = _resolve_vision_tower(model)
    blocks = vision_model.encoder.layer
    cfg = vision_model.config
    hidden_size = cfg.hidden_size
    num_heads = host_num_heads(cfg)
    activation = getattr(cfg, "hidden_act", "gelu") or "gelu"
    norm_kind = _norm_kind(getattr(cfg, "norm_type", None))
    qk_norm = bool(getattr(cfg, "use_qk_norm", False))
    p = next(vision_model.parameters())

    for block in blocks:  # inject every ViT block (paper recipe)
        sve_blk = make_sve_block(hidden_size, num_heads, cfg.intermediate_size,
                       proj_init_std=proj_init_std, no_pos_embed=True,
                       dtype=p.dtype, device=p.device,
                       ffn_kind="gelu_fc1_fc2", activation=activation,
                       norm_kind=norm_kind, qk_norm=qk_norm)
        init_block_from_self_attn(sve_blk, block, source_layout="internvl")
        block.sve_block = sve_blk
        block._sve_ctx = None
        block.forward = types.MethodType(_layer_forward, block)

    register_seqlens_routing(model, vision_model)
    register_dense_pre_hook(vision_model, blocks)
    freeze_except_sve(blocks)
    persist_sve_config(model)
    return count_sve_params(blocks)
