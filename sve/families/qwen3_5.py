"""SVE injection for Qwen3.5 (``model_type="qwen3_5"``).

Vision encoder at ``model.model.visual``; blocks at ``.blocks`` are
``Qwen3_5VisionBlock`` with fused-QKV self-attn, LayerNorms, a classic 2-Linear
GELU MLP, and vision-RoPE on Q/K. The stock vision forward already passes
``cu_seqlens`` and ``position_embeddings`` to each block, so we inject via a
vision pre-hook + per-block monkey-patch (no vision-forward replacement).
"""

import types

from ..utils import (
    count_sve_params, ddp_touch, freeze_except_sve, host_num_heads,
    init_block_from_self_attn, make_sve_block, persist_sve_config,
    register_seqlens_routing, register_varlen_pre_hook, sve_varlen_pre,
)


def _block_forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
    hidden_states = sve_varlen_pre(self, hidden_states, cu_seqlens, position_embeddings)
    hidden_states = hidden_states + self.attn(
        self.norm1(hidden_states), cu_seqlens=cu_seqlens,
        position_embeddings=position_embeddings, **kwargs,
    )
    hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
    return hidden_states + ddp_touch(self)


def inject(model, *, proj_init_std: float) -> int:
    vision_model = model.model.visual
    blocks = vision_model.blocks
    hidden_size = vision_model.config.hidden_size
    num_heads = host_num_heads(vision_model.config)
    activation = getattr(vision_model.config, "hidden_act", "gelu_pytorch_tanh") or "gelu_pytorch_tanh"
    p = next(vision_model.parameters())

    for block in blocks:  # inject every ViT block (paper recipe)
        intermediate = block.mlp.linear_fc1.weight.shape[0]
        sve_blk = make_sve_block(hidden_size, num_heads, intermediate,
                       proj_init_std=proj_init_std, no_pos_embed=False,
                       dtype=p.dtype, device=p.device,
                       ffn_kind="gelu_fc1_fc2", activation=activation, norm_kind="layer_norm")
        init_block_from_self_attn(sve_blk, block, source_layout="qwen35_flat")
        block.sve_block = sve_blk
        block._sve_ctx = None
        block.forward = types.MethodType(_block_forward, block)

    register_seqlens_routing(model, vision_model)
    register_varlen_pre_hook(vision_model, blocks)
    freeze_except_sve(blocks)
    persist_sve_config(model)
    return count_sve_params(blocks)
