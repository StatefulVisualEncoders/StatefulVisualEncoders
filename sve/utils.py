"""
Shared plumbing for the per-family SVE injection adapters.

All five family adapters follow the same shape:
  1. Locate the ViT block list on the model.
  2. Build a ``CrossFFNBlock`` sized to the family's vision hidden dim
     and structured to mirror the family's block (FFN/norm kind, activation).
  3. Clone Q/K/V (+ FFN first linear) from the block's self-attention via
     ``init_block_from_self_attn`` (output projections stay zero/small-init).
  4. Monkey-patch ``block.forward`` to run the SVE block *before* SA+FFN.
  5. Route the per-sample image grouping (``image_seqlens_per_sample``) from the
     batch into the vision forward via an FSDP-safe thread-local + hooks.

The cross-frame KV is assembled per ViT layer: for image ``i`` the keys/values
come from image ``i-1`` (stop-gradient'd), with the first image of each sample
falling back to attending to itself (``Z1`` fallback).
"""

import threading
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_ffn import CrossFFNBlock


# Thread-local routing of image_seqlens_per_sample (FSDP-safe, per-rank).
_ctx = threading.local()


def _set_seqlens(value: Optional[List[int]]) -> None:
    _ctx.image_seqlens_per_sample = value


def _get_seqlens() -> Optional[List[int]]:
    return getattr(_ctx, "image_seqlens_per_sample", None)


def _clear_seqlens() -> None:
    _ctx.image_seqlens_per_sample = None


def host_num_heads(cfg) -> int:
    """The SVE block always uses the host vision encoder's head count, so its
    head_dim matches the host (required for host RoPE cos/sin to broadcast over the
    the block's Q/K). This is fixed by the architecture — not a hyperparameter."""
    return getattr(cfg, "num_heads", None) or cfg.num_attention_heads


def build_prev_img_map(image_seqlens_per_sample: List[int], n_images: int) -> dict:
    """Map each image index to its predecessor (or None for a sample's first).

    image_seqlens_per_sample=[3,1,2] -> {0:None,1:0,2:1, 3:None, 4:None,5:4}.
    """
    prev: dict = {}
    offset = 0
    for n in image_seqlens_per_sample:
        for pos in range(n):
            i = offset + pos
            prev[i] = (offset + pos - 1) if pos > 0 else None
        offset += n
    for i in range(n_images):
        prev.setdefault(i, None)
    return prev


def make_sve_block(
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
    *,
    proj_init_std: float,
    no_pos_embed: bool,
    dtype: torch.dtype,
    device: torch.device,
    ffn_kind: str = "gelu_fc1_fc2",
    activation: str = "gelu_tanh",
    norm_kind: str = "layer_norm",
    qk_norm: bool = False,
    ffn_bias: bool = True,
) -> CrossFFNBlock:
    """Build a SVE block mirroring the host family's vision block, on the
    model's dtype/device."""
    sve_blk = CrossFFNBlock(
        hidden_size, num_heads, intermediate_size,
        proj_init_std=proj_init_std,
        no_pos_embed=no_pos_embed,
        ffn_kind=ffn_kind,
        activation=activation,
        norm_kind=norm_kind,
        qk_norm=qk_norm,
        ffn_bias=ffn_bias,
    )
    return sve_blk.to(dtype=dtype, device=device)


def init_block_from_self_attn(sve_blk: CrossFFNBlock, block: nn.Module, *, source_layout: str) -> None:
    """Clone Q/K/V, input/pre-FFN norms, and the FFN's first linear from a
    self-attention block into the SVE block. Output projections (``proj``
    and the FFN second linear) are left at their zero/small init so the module
    starts as a no-op contribution.

    ``source_layout`` selects how to read the host block:
      - "qwen35_flat": fused ``block.attn.qkv`` (bias=True), LayerNorms
        ``block.norm1/norm2``, classic MLP ``linear_fc1/linear_fc2``
        (Qwen3.5 and Qwen3-VL).
      - "glm4v_swiglu": fused ``block.attn.qkv``, RMSNorms, SwiGLU MLP
        (``gate_proj``/``up_proj``/``down_proj``), bias-free.
      - "internvl": separate ``block.attention.{q,k,v}_proj`` + norms
        ``layernorm_before/after``, classic MLP ``fc1/fc2``.
      - "siglip": separate ``block.self_attn.{q,k,v}_proj`` + norms
        ``layer_norm1/layer_norm2``, classic MLP ``fc1/fc2`` (Gemma-3).
    """
    hidden_size = sve_blk.hidden_size

    def _copy_(dst, src):
        dst.data.copy_(src.to(dst.dtype))

    def _copy_norm(dst_norm, src_norm):
        sw = getattr(src_norm, "weight", None)
        if sw is not None:
            _copy_(dst_norm.weight, sw.data)
        sb = getattr(src_norm, "bias", None)
        dst_bias = getattr(dst_norm, "bias", None)
        if dst_bias is not None:
            dst_bias.data.copy_(sb.data.to(dst_bias.dtype)) if sb is not None else dst_bias.data.zero_()

    def _clone_fc1(src_fc1: nn.Linear):
        _copy_(sve_blk.fc1.weight, src_fc1.weight.data)
        if sve_blk.fc1.bias is not None:
            sve_blk.fc1.bias.data.copy_(src_fc1.bias.data.to(sve_blk.fc1.bias.dtype)) if src_fc1.bias is not None else sve_blk.fc1.bias.data.zero_()

    def _clone_swiglu_in(src_gate: nn.Linear, src_up: nn.Linear):
        _copy_(sve_blk.gate_proj.weight, src_gate.weight.data)
        _copy_(sve_blk.up_proj.weight, src_up.weight.data)
        for dst_lin, src_lin in ((sve_blk.gate_proj, src_gate), (sve_blk.up_proj, src_up)):
            if dst_lin.bias is not None:
                dst_lin.bias.data.copy_(src_lin.bias.data.to(dst_lin.bias.dtype)) if getattr(src_lin, "bias", None) is not None else dst_lin.bias.data.zero_()

    with torch.no_grad():
        if source_layout in ("qwen35_flat", "glm4v_swiglu"):
            base_qkv_w = block.attn.qkv.weight.data
            base_qkv_b = getattr(block.attn.qkv, "bias", None)
            _copy_(sve_blk.qkv.weight, base_qkv_w)
            sve_blk.qkv.bias.data.copy_(base_qkv_b.data.to(sve_blk.qkv.bias.dtype)) if base_qkv_b is not None else sve_blk.qkv.bias.data.zero_()
            _copy_norm(sve_blk.norm1_q, block.norm1)
            _copy_norm(sve_blk.norm1_kv, block.norm1)
            _copy_norm(sve_blk.norm2, block.norm2)
            if source_layout == "qwen35_flat":
                _clone_fc1(block.mlp.linear_fc1)
            else:
                _clone_swiglu_in(block.mlp.gate_proj, block.mlp.up_proj)

        elif source_layout in ("internvl", "siglip"):
            attn = block.attention if source_layout == "internvl" else block.self_attn
            _copy_(sve_blk.qkv.weight[:hidden_size], attn.q_proj.weight.data)
            _copy_(sve_blk.qkv.weight[hidden_size:2 * hidden_size], attn.k_proj.weight.data)
            _copy_(sve_blk.qkv.weight[2 * hidden_size:], attn.v_proj.weight.data)
            sve_blk.qkv.bias.data.zero_()
            for src, slc in (
                (attn.q_proj, slice(0, hidden_size)),
                (attn.k_proj, slice(hidden_size, 2 * hidden_size)),
                (attn.v_proj, slice(2 * hidden_size, 3 * hidden_size)),
            ):
                if src.bias is not None:
                    _copy_(sve_blk.qkv.bias[slc], src.bias.data)
            if source_layout == "internvl":
                _copy_norm(sve_blk.norm1_q, block.layernorm_before)
                _copy_norm(sve_blk.norm1_kv, block.layernorm_before)
                _copy_norm(sve_blk.norm2, block.layernorm_after)
            else:
                _copy_norm(sve_blk.norm1_q, block.layer_norm1)
                _copy_norm(sve_blk.norm1_kv, block.layer_norm1)
                _copy_norm(sve_blk.norm2, block.layer_norm2)
            _clone_fc1(block.mlp.fc1)
        else:
            raise ValueError(f"Unknown source_layout={source_layout!r}")


def assemble_qkv_pairs_with_cu(
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prev_img_map: dict,
    *,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    null_embedding: Optional[torch.Tensor],
    use_null_embed: bool,
    kv_stop_grad: bool,
):
    """Build flat (Q, KV, cu_q, cu_kv, max_q, max_kv, pos_q, pos_kv) for batched
    cross-attention. Image i's KV is image (i-1)'s tokens (detached when
    kv_stop_grad); a sample's first image attends to itself (Z1 fallback)."""
    n_images = len(cu_seqlens) - 1
    cos_full = sin_full = None
    if position_embeddings is not None:
        cos_full, sin_full = position_embeddings

    q_chunks, kv_chunks = [], []
    q_pos_chunks, kv_pos_chunks = [], []
    q_seqlens, kv_seqlens = [0], [0]

    for i in range(n_images):
        qs, qe = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
        q_chunks.append(hidden_states[qs:qe])
        q_seqlens.append(q_seqlens[-1] + (qe - qs))
        if cos_full is not None:
            q_pos_chunks += [cos_full[qs:qe], sin_full[qs:qe]]

        pred = prev_img_map.get(i)
        if pred is None:
            if use_null_embed and null_embedding is not None:
                kv_chunks.append(null_embedding)
                kv_seqlens.append(kv_seqlens[-1] + 1)
                if cos_full is not None:
                    kv_pos_chunks += [cos_full[qs:qs + 1], sin_full[qs:qs + 1]]
            else:
                # Z1 fallback: first image attends to itself.
                kv_chunks.append(hidden_states[qs:qe])
                kv_seqlens.append(kv_seqlens[-1] + (qe - qs))
                if cos_full is not None:
                    kv_pos_chunks += [cos_full[qs:qe], sin_full[qs:qe]]
        else:
            ks, ke = int(cu_seqlens[pred].item()), int(cu_seqlens[pred + 1].item())
            kv = hidden_states[ks:ke]
            if kv_stop_grad:
                kv = kv.detach()
            kv_chunks.append(kv)
            kv_seqlens.append(kv_seqlens[-1] + (ke - ks))
            if cos_full is not None:
                kv_pos_chunks += [cos_full[ks:ke], sin_full[ks:ke]]

    q_flat = torch.cat(q_chunks, dim=0)
    kv_flat = torch.cat(kv_chunks, dim=0)
    cu_q = torch.tensor(q_seqlens, dtype=torch.int32, device=hidden_states.device)
    cu_kv = torch.tensor(kv_seqlens, dtype=torch.int32, device=hidden_states.device)
    max_q = max(q_seqlens[i + 1] - q_seqlens[i] for i in range(len(q_seqlens) - 1))
    max_kv = max(kv_seqlens[i + 1] - kv_seqlens[i] for i in range(len(kv_seqlens) - 1))

    pos_q = pos_kv = None
    if cos_full is not None:
        pos_q = (torch.cat(q_pos_chunks[0::2], dim=0), torch.cat(q_pos_chunks[1::2], dim=0))
        pos_kv = (torch.cat(kv_pos_chunks[0::2], dim=0), torch.cat(kv_pos_chunks[1::2], dim=0))
    return q_flat, kv_flat, cu_q, cu_kv, max_q, max_kv, pos_q, pos_kv


def assemble_qkv_pairs_dense(
    hidden_states: torch.Tensor,
    prev_img_map: dict,
    *,
    null_embedding: Optional[torch.Tensor],
    use_null_embed: bool,
    kv_stop_grad: bool,
):
    """Dense [B, S, D] variant for families that loop over images (InternVL,
    SigLIP/Gemma-3). Returns parallel per-image (q, kv) lists."""
    if hidden_states.dim() != 3:
        raise ValueError("expected [B, S, D]")
    n_images = hidden_states.shape[0]
    q_list, kv_list = [], []
    for i in range(n_images):
        q_list.append(hidden_states[i])
        pred = prev_img_map.get(i)
        if pred is None:
            if use_null_embed and null_embedding is not None:
                kv_list.append(null_embedding)
            else:
                kv_list.append(hidden_states[i])  # Z1 fallback
        else:
            kv = hidden_states[pred]
            kv_list.append(kv.detach() if kv_stop_grad else kv)
    return q_list, kv_list


def register_seqlens_routing(model: nn.Module, vision_model: nn.Module) -> None:
    """Route image_seqlens_per_sample from the outer batch into the vision
    forward via a per-rank thread-local (FSDP-safe)."""
    def _outer_pre_hook(module, args, kwargs):
        _set_seqlens(kwargs.pop("image_seqlens_per_sample", None))
        return args, kwargs

    model.register_forward_pre_hook(_outer_pre_hook, with_kwargs=True)

    def _vision_pre_hook(module, args, kwargs):
        seqlens = _get_seqlens()
        if seqlens is not None:
            kwargs["image_seqlens_per_sample"] = seqlens
        return args, kwargs

    vision_model.register_forward_pre_hook(_vision_pre_hook, with_kwargs=True)
    vision_model.register_forward_hook(lambda module, inp, out: _clear_seqlens())


def freeze_except_sve(blocks: Iterable[nn.Module]) -> None:
    """Ensure SVE-block params are trainable (init_adapter may have frozen
    everything when only the SVE is being trained)."""
    for blk in blocks:
        sve_blk = getattr(blk, "sve_block", None)
        if sve_blk is not None:
            for p in sve_blk.parameters():
                p.requires_grad_(True)


def count_sve_params(blocks: Iterable[nn.Module]) -> int:
    n = 0
    for blk in blocks:
        sve_blk = getattr(blk, "sve_block", None)
        if sve_blk is not None:
            n += sum(p.numel() for p in sve_blk.parameters())
    return n


def persist_sve_config(model: nn.Module) -> None:
    """Mark the model as SVE-injected on its config so save_pretrained records it
    in config.json (read back at inference to re-inject the same structure before
    loading weights). The recipe is fixed, so a single flag suffices."""
    model.config.sve_enable = True


# ---------------------------------------------------------------------------
# Runtime: per-ViT-layer cross-image conditioning. Two shapes:
#   * varlen — Qwen3.5 / Qwen3-VL / GLM-4.6V: flat patch tokens + cu_seqlens,
#     batched via flash-varlen. RoPE on Q/K.
#   * dense  — InternVL / SigLIP(Gemma-3): [B=n_images, S, D] tensors, per-image
#     loop. No RoPE (absolute positions at patch-embed).
# A pre-hook on the vision model builds the per-call context (cu_seqlens +
# predecessor map) and stashes it on each injected block as ``_sve_ctx``; the
# monkey-patched block forward reads it and runs the SVE block before SA+FFN.
# The final recipe is fixed: use_null_embed=False (Z1 self-fallback),
# kv_stop_grad=True (detach predecessor keys/values).
# ---------------------------------------------------------------------------


def _consume_eval_seqlens(module: nn.Module) -> Optional[List[int]]:
    """Eval-time fallback: ``model.generate`` can't thread image_seqlens through,
    so the eval harness sets ``vision_model._eval_image_seqlens`` before forward."""
    s = getattr(module, "_eval_image_seqlens", None)
    if s is not None:
        module._eval_image_seqlens = None  # consume once
    return s


def _make_varlen_ctx(grid_thw: torch.Tensor, seqlens: List[int]):
    """Recompute cu_seqlens exactly as the host vision forward does, and the
    predecessor map. Returns (cu_seqlens, prev_img_map)."""
    cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu = F.pad(cu, (1, 0), value=0)
    return cu, build_prev_img_map(seqlens, len(cu) - 1)


def register_varlen_pre_hook(vision_model: nn.Module, blocks) -> None:
    """Hook on a flat-token vision model (forward(hidden_states, grid_thw, ...))
    that stashes (cu_seqlens, prev_img_map) on each injected block."""
    def hook(module, args, kwargs):
        seqlens = _get_seqlens() or _consume_eval_seqlens(module)
        grid_thw = args[1] if len(args) > 1 else kwargs.get("grid_thw")
        if seqlens is None or grid_thw is None:
            for b in blocks:
                b._sve_ctx = None
            return args, kwargs
        ctx = _make_varlen_ctx(grid_thw, seqlens)
        for b in blocks:
            b._sve_ctx = ctx if getattr(b, "sve_block", None) is not None else None
        return args, kwargs

    vision_model.register_forward_pre_hook(hook, with_kwargs=True)


def register_dense_pre_hook(vision_model: nn.Module, blocks) -> None:
    """Hook on a dense [B,S,D] vision model (forward(pixel_values, ...)) that
    stashes prev_img_map on each injected block."""
    def hook(module, args, kwargs):
        seqlens = _get_seqlens() or _consume_eval_seqlens(module)
        pv = args[0] if args else kwargs.get("pixel_values")
        if seqlens is None or pv is None:
            for b in blocks:
                b._sve_ctx = None
            return args, kwargs
        prev = build_prev_img_map(seqlens, pv.shape[0])
        for b in blocks:
            b._sve_ctx = prev if getattr(b, "sve_block", None) is not None else None
        return args, kwargs

    vision_model.register_forward_pre_hook(hook, with_kwargs=True)


def sve_varlen_pre(block, hidden_states, cu_seqlens, position_embeddings, rope_apply_fn=None):
    """Run the SVE block over flat tokens before the host block's SA+FFN.
    Returns hidden_states with current-image tokens replaced by their cross-image
    output (a no-op at init because output projections are zero-initialized)."""
    ctx = getattr(block, "_sve_ctx", None)
    if ctx is None:
        return hidden_states
    cu_ctx, prev_map = ctx
    # The pre-hook's cu_seqlens and the block's forward-arg cu_seqlens both derive
    # deterministically from grid_thw; guard against a silent transformers drift.
    assert torch.equal(cu_ctx, cu_seqlens), "cu_seqlens mismatch: prehook vs forward arg"
    q_flat, kv_flat, cu_q, cu_kv, max_q, max_kv, pos_q, pos_kv = assemble_qkv_pairs_with_cu(
        hidden_states, cu_ctx, prev_map,
        position_embeddings=position_embeddings,
        null_embedding=None, use_null_embed=False, kv_stop_grad=True,
    )
    out = block.sve_block.forward_batched(
        q_flat, kv_flat, cu_q, cu_kv, max_q, max_kv,
        position_embeddings_q=pos_q, position_embeddings_kv=pos_kv,
        rope_apply_fn=rope_apply_fn,
    )
    hidden_states = hidden_states.clone()
    for i in range(len(cu_ctx) - 1):
        qs, qe = int(cu_ctx[i].item()), int(cu_ctx[i + 1].item())
        cs, ce = int(cu_q[i].item()), int(cu_q[i + 1].item())
        hidden_states[qs:qe] = out[cs:ce]
    return hidden_states


def sve_dense_pre(block, hidden_states):
    """Run the SVE block over dense [B,S,D] tokens before the host block's
    SA+FFN (per-image loop; no RoPE)."""
    ctx = getattr(block, "_sve_ctx", None)
    if ctx is None or hidden_states.dim() != 3:
        return hidden_states
    q_list, kv_list = assemble_qkv_pairs_dense(
        hidden_states, ctx, null_embedding=None, use_null_embed=False, kv_stop_grad=True,
    )
    out = [block.sve_block(q, prev_tokens=kv, position_embeddings=None)
           for q, kv in zip(q_list, kv_list)]
    return torch.stack(out, dim=0)


def ddp_touch(block):
    """Zero-valued sum over all SVE params so every param participates in the
    graph each step (keeps DDP/FSDP happy when a param is conditionally unused)."""
    return sum((p * 0).sum() for p in block.sve_block.parameters())
