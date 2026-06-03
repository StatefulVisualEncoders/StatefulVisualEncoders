"""
Stateful Visual Encoder (SVE) — Cross+FFN block.

This is the single learnable module the SVE adds to a pretrained VLM's vision
encoder. It implements the paper's winning **Cross+FFN** design: for each ViT
block we insert, *before* the block's self-attention, a cross-attention layer
whose queries come from the current image and whose keys/values come from the
previous image, followed by a feed-forward network. The block form is:

    x  = x + proj( CrossAttn( Q = norm1_q(x), K,V = norm1_kv(prev) ) )
    x  = x + FFN( norm2(x) )

structurally mirroring the host family's vision block so its weights can be
cloned 1:1 at init (see ``sve.utils.init_block_from_self_attn``).

Family parameterization
-----------------------
The same module backs all five VLM families in the paper by selecting:
  * ``ffn_kind``  — ``"gelu_fc1_fc2"`` (Qwen3.5 / Qwen3-VL / InternVL / SigLIP)
                    or ``"swiglu_gate_up_down"`` (GLM-4.6V).
  * ``norm_kind`` — ``"layer_norm"`` (default) or ``"rms_norm"`` (GLM-4.6V).
  * ``qk_norm``   — per-head RMSNorm on Q/K (InternVL when ``use_qk_norm``).
  * ``no_pos_embed`` — drop RoPE on Q/K for families whose host self-attn uses
                    learned absolute positions instead (InternVL, SigLIP/Gemma-3).

Recipe defaults (paper §3.2/§3.4)
---------------------------------
  * Output projections (``proj`` and the FFN's second linear) are **zero-init**
    so the module starts as an exact no-op, preserving the pretrained encoder's
    feature distribution at the start of finetuning. Real-world tasks instead
    use a tiny ``proj_init_std = 1e-4`` after ablations showed slightly better
    optimization.
  * Q/K/V and the FFN's first linear are cloned from the host block at init.
  * Keys/values from the previous image are stop-gradient'd by the caller.
"""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


# Activation registry — string -> nn.Module factory. Adapters pass a string
# matching the host's ``config.hidden_act`` so the cloned FFN is structurally
# identical to the source.
_ACT_FACTORIES = {
    "gelu_tanh":          lambda: nn.GELU(approximate="tanh"),
    "gelu_pytorch_tanh":  lambda: nn.GELU(approximate="tanh"),
    "gelu":               lambda: nn.GELU(),
    "gelu_new":           lambda: nn.GELU(approximate="tanh"),
    "silu":               lambda: nn.SiLU(),
    "swish":              lambda: nn.SiLU(),
    "relu":               lambda: nn.ReLU(),
}


def _make_activation(name: str) -> nn.Module:
    if name not in _ACT_FACTORIES:
        raise ValueError(f"Unknown activation {name!r}. Known: {sorted(_ACT_FACTORIES)}")
    return _ACT_FACTORIES[name]()


class _RMSNorm(nn.Module):
    """Weight-only RMSNorm (no bias). Local impl so this module doesn't depend on
    which transformers version / family RMSNorm class is loaded."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight).to(orig_dtype)


def _make_norm(kind: str, hidden_size: int, eps: float = 1e-6) -> nn.Module:
    if kind == "layer_norm":
        return nn.LayerNorm(hidden_size, eps=eps)
    if kind == "rms_norm":
        return _RMSNorm(hidden_size, eps=eps)
    raise ValueError(f"Unknown norm_kind={kind!r}")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor, cos, sin):
    """Apply rotary position embedding. Shape: [S, H, D]."""
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    out = (tensor * cos) + (rotate_half(tensor) * sin)
    return out.to(orig_dtype)


class CrossFFNBlock(nn.Module):
    """Cross-attention + FFN SVE block (the SVE's only learnable module).

    Structurally mirrors the host family's vision block so weights can be cloned
    1:1. See module docstring for the block form and family parameterization.

    Args:
        hidden_size: token dim.
        num_heads: attention heads (must give the same head_dim as the host so
            host RoPE cos/sin broadcast correctly over the block's Q/K).
        intermediate_size: FFN inner dim (= host block.mlp's intermediate dim).
        proj_init_std: if > 0, init output projections with ``normal_(0, std)``
            (real-world recipe, 1e-4); otherwise zero-init (synthetic recipe).
        no_pos_embed: never apply RoPE to Q/K (families whose host self-attn uses
            learned absolute positions: InternVL, SigLIP/Gemma-3).
        ffn_kind: ``"gelu_fc1_fc2"`` (default) or ``"swiglu_gate_up_down"``.
        activation: name in ``_ACT_FACTORIES``; mirror the host MLP's activation.
        norm_kind: ``"layer_norm"`` (default) or ``"rms_norm"`` (GLM-4.6V).
        qk_norm: add per-head RMSNorm on Q and K (InternVL's optional qk_norm).
        ffn_bias: whether FFN linears have bias (``False`` for GLM-4.6V SwiGLU).
    """

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int = 4096,
                 proj_init_std: float = 0.0,
                 no_pos_embed: bool = False,
                 ffn_kind: str = "gelu_fc1_fc2",
                 activation: str = "gelu_tanh",
                 norm_kind: str = "layer_norm",
                 qk_norm: bool = False,
                 ffn_bias: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.no_pos_embed = no_pos_embed
        self.ffn_kind = ffn_kind
        self.activation_name = activation
        self.norm_kind = norm_kind
        self.qk_norm = qk_norm
        self.ffn_bias = ffn_bias

        # --- Norms (mirror host block.norm1/norm2 by kind) ---
        self.norm1_q = _make_norm(norm_kind, hidden_size)
        self.norm1_kv = _make_norm(norm_kind, hidden_size)

        # Cross-attention QKV (fused — same shape the host block.attn.qkv has).
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # Optional QK-norm (InternVL when config.use_qk_norm is True).
        if qk_norm:
            self.q_norm = _make_norm("rms_norm", hidden_size)
            self.k_norm = _make_norm("rms_norm", hidden_size)
        else:
            self.q_norm = None
            self.k_norm = None

        # --- FFN (family-native: classic 2-Linear GELU, or SwiGLU) ---
        self.norm2 = _make_norm(norm_kind, hidden_size)
        self.act = _make_activation(activation)
        if ffn_kind == "gelu_fc1_fc2":
            self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
            self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=ffn_bias)
            self.gate_proj = self.up_proj = self.down_proj = None  # type: ignore[assignment]
        elif ffn_kind == "swiglu_gate_up_down":
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=ffn_bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=ffn_bias)
            if ffn_bias:
                nn.init.zeros_(self.gate_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
                nn.init.zeros_(self.down_proj.bias)
            self.fc1 = self.fc2 = None  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown ffn_kind={ffn_kind!r}")

        # Null embedding for the first frame (used only when the caller opts into
        # a learned null KV instead of the Z1 self-fallback; kept for checkpoint
        # compatibility).
        self.null_embedding = nn.Parameter(torch.empty(1, hidden_size))
        nn.init.normal_(self.null_embedding, std=0.02)

        # Init output projections: tiny-normal (proj_init_std>0) or zero (default),
        # so the module starts as a no-op (or near-no-op) contribution.
        out = self._ffn_out_linear()
        if proj_init_std > 0:
            nn.init.normal_(self.proj.weight, std=proj_init_std)
            nn.init.zeros_(self.proj.bias)
            nn.init.normal_(out.weight, std=proj_init_std)
            if out.bias is not None:
                nn.init.zeros_(out.bias)
        else:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)
            nn.init.zeros_(out.weight)
            if out.bias is not None:
                nn.init.zeros_(out.bias)

    # ---- FFN helpers --------------------------------------------------------

    def _ffn_out_linear(self) -> nn.Linear:
        """Return the FFN's output projection (zero-initialized so the SVE block starts as
        a no-op contribution)."""
        if self.ffn_kind == "gelu_fc1_fc2":
            return self.fc2
        return self.down_proj

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_kind == "gelu_fc1_fc2":
            return self.fc2(self.act(self.fc1(x)))
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))

    # ---- Forward (dense per-pair; used by InternVL / SigLIP adapters) --------

    def forward(
        self,
        tokens: torch.Tensor,                         # [S, D] current frame
        prev_tokens: Optional[torch.Tensor] = None,   # [S', D] previous frame
        use_null_embed: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        kv_position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_apply_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        if prev_tokens is None:
            if use_null_embed:
                prev_tokens = self.null_embedding
            else:
                return tokens

        S = tokens.shape[0]
        S_prev = prev_tokens.shape[0]

        q_in = self.norm1_q(tokens)
        kv_in = self.norm1_kv(prev_tokens)

        q = self.qkv(q_in)[:, :self.hidden_size].view(S, self.num_heads, self.head_dim)
        kv_out = self.qkv(kv_in)
        k = kv_out[:, self.hidden_size:2 * self.hidden_size].view(S_prev, self.num_heads, self.head_dim)
        v = kv_out[:, 2 * self.hidden_size:].view(S_prev, self.num_heads, self.head_dim)

        if self.q_norm is not None:
            q = self.q_norm(q.view(S, self.hidden_size)).view(S, self.num_heads, self.head_dim)
            k = self.k_norm(k.view(S_prev, self.hidden_size)).view(S_prev, self.num_heads, self.head_dim)

        if position_embeddings is not None and not self.no_pos_embed:
            cos_q, sin_q = position_embeddings
            apply = rope_apply_fn or apply_rotary_pos_emb_vision
            q = apply(q, cos_q, sin_q)
            cos_k, sin_k = kv_position_embeddings if kv_position_embeddings is not None else (cos_q, sin_q)
            if S_prev == cos_k.shape[0]:
                k = apply(k, cos_k, sin_k)

        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.squeeze(0).transpose(0, 1).contiguous().view(S, -1)

        tokens = tokens + self.proj(attn_out)
        tokens = tokens + self._ffn(self.norm2(tokens))
        return tokens

    # ---- Batched forward (flash-varlen; used by Qwen3.5/3-VL/GLM adapters) ---

    def forward_batched(
        self,
        hidden_states: torch.Tensor,           # [total_q_tokens, D] flat Q tokens
        kv_hidden_states: torch.Tensor,        # [total_kv_tokens, D] flat KV tokens
        cu_seqlens_q: torch.Tensor,            # [n_pairs+1] cumulative Q seq lengths
        cu_seqlens_kv: torch.Tensor,           # [n_pairs+1] cumulative KV seq lengths
        max_seqlen_q: int,
        max_seqlen_kv: int,
        position_embeddings_q: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_embeddings_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        rope_apply_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """Batched cross-attention using ``flash_attn_varlen_func`` (Q chunk i
        attends to KV chunk i). Falls back to a per-pair ``forward`` loop when
        flash-attn is unavailable."""
        if not HAS_FLASH_ATTN:
            n_pairs = len(cu_seqlens_q) - 1
            outputs = []
            for i in range(n_pairs):
                q_tokens = hidden_states[cu_seqlens_q[i]:cu_seqlens_q[i + 1]]
                kv_tokens = kv_hidden_states[cu_seqlens_kv[i]:cu_seqlens_kv[i + 1]]
                q_pos = kv_pos = None
                if position_embeddings_q is not None:
                    cos_q, sin_q = position_embeddings_q
                    q_pos = (cos_q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]],
                             sin_q[cu_seqlens_q[i]:cu_seqlens_q[i + 1]])
                if position_embeddings_kv is not None:
                    cos_kv, sin_kv = position_embeddings_kv
                    kv_pos = (cos_kv[cu_seqlens_kv[i]:cu_seqlens_kv[i + 1]],
                              sin_kv[cu_seqlens_kv[i]:cu_seqlens_kv[i + 1]])
                outputs.append(self.forward(q_tokens, kv_tokens,
                                            position_embeddings=q_pos,
                                            kv_position_embeddings=kv_pos,
                                            rope_apply_fn=rope_apply_fn))
            return torch.cat(outputs, dim=0)

        q_in = self.norm1_q(hidden_states)
        kv_in = self.norm1_kv(kv_hidden_states)

        q_all = self.qkv(q_in)[:, :self.hidden_size].view(-1, self.num_heads, self.head_dim)
        kv_out = self.qkv(kv_in)
        k_all = kv_out[:, self.hidden_size:2 * self.hidden_size].view(-1, self.num_heads, self.head_dim)
        v_all = kv_out[:, 2 * self.hidden_size:].view(-1, self.num_heads, self.head_dim)

        if self.q_norm is not None:
            q_all = self.q_norm(q_all.view(-1, self.hidden_size)).view(-1, self.num_heads, self.head_dim)
            k_all = self.k_norm(k_all.view(-1, self.hidden_size)).view(-1, self.num_heads, self.head_dim)

        apply = rope_apply_fn or apply_rotary_pos_emb_vision
        if position_embeddings_q is not None and not self.no_pos_embed:
            cos_q, sin_q = position_embeddings_q
            q_all = apply(q_all, cos_q, sin_q)
        if position_embeddings_kv is not None and not self.no_pos_embed:
            cos_k, sin_k = position_embeddings_kv
            k_all = apply(k_all, cos_k, sin_k)

        attn_out = flash_attn_varlen_func(
            q_all.to(torch.bfloat16),
            k_all.to(torch.bfloat16),
            v_all.to(torch.bfloat16),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_kv,
            causal=False,
        ).to(hidden_states.dtype)
        attn_out = attn_out.view(-1, self.hidden_size)

        result = hidden_states + self.proj(attn_out)
        result = result + self._ffn(self.norm2(result))
        return result
