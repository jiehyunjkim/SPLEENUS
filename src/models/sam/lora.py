"""LoRA (Low-Rank Adaptation) for the native `sam3` model.

This is a cleaned-up version of the vendored `lora_layers.py`. Behavior is
unchanged on purpose: the pretrained MedSAM3 LoRA checkpoint
(`lal-Joey/MedSAM3_v1`) was trained against this exact module-replacement
scheme, so the *names* and *shapes* of the LoRA-wrapped submodules must stay
identical or `load_lora_weights(..., strict=False)` will silently fail to
load most of the checkpoint.

Two things make this file more than "wrap every nn.Linear":
1. `nn.MultiheadAttention` fuses Q/K/V into one `in_proj_weight`, so LoRA
   can't be attached to Q/K/V separately. `MultiheadAttentionLoRA` splits it
   into three plain `nn.Linear` layers first, copying the original weights,
   so each one becomes LoRA-able.
2. Which components get adapted (vision encoder, text encoder, geometry
   encoder, DETR encoder/decoder, mask decoder) is controlled per-component
   through `LoRAConfig`, matching how MedSAM3 was actually trained.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Submodule name fragments that receive LoRA by default. Covers both
# separate-projection attention (q_proj/k_proj/v_proj/out_proj) and the
# fused qkv projection used in the ViT-style vision backbone, plus the MLP
# layers in each backbone flavor.
DEFAULT_TARGET_MODULES = {
    "q_proj", "k_proj", "v_proj", "out_proj",       # standard attention
    "qkv", "proj",                                   # ViT-style vision backbone
    "fc1", "fc2",                                    # vision backbone MLP
    "c_fc", "c_proj",                                # CLIP-style text backbone MLP
    "linear1", "linear2",                            # transformer encoder/decoder FFN
}


class MultiheadAttentionLoRA(nn.Module):
    """Drop-in replacement for `nn.MultiheadAttention` with separate Q/K/V
    projections, so LoRA can be attached to each one individually.

    Constructed from an existing `nn.MultiheadAttention`'s weights so the
    replacement is numerically identical before any LoRA is applied.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        in_proj_weight: Optional[torch.Tensor] = None,
        in_proj_bias: Optional[torch.Tensor] = None,
        out_proj_weight: Optional[torch.Tensor] = None,
        out_proj_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if in_proj_weight is not None:
            self.q_proj.weight.data = in_proj_weight[:embed_dim].clone()
            self.k_proj.weight.data = in_proj_weight[embed_dim : 2 * embed_dim].clone()
            self.v_proj.weight.data = in_proj_weight[2 * embed_dim :].clone()
        if in_proj_bias is not None:
            self.q_proj.bias.data = in_proj_bias[:embed_dim].clone()
            self.k_proj.bias.data = in_proj_bias[embed_dim : 2 * embed_dim].clone()
            self.v_proj.bias.data = in_proj_bias[2 * embed_dim :].clone()
        if out_proj_weight is not None:
            self.out_proj.weight.data = out_proj_weight.clone()
        if out_proj_bias is not None:
            self.out_proj.bias.data = out_proj_bias.clone()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.batch_first:
            bsz, tgt_len, _ = query.shape
            src_len = key.shape[1]
        else:
            tgt_len, bsz, _ = query.shape
            src_len = key.shape[0]
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))

        q = self.q_proj(query).view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                if attn_mask.shape[0] == bsz:
                    attn_mask = attn_mask.unsqueeze(1)
                elif attn_mask.shape[0] == bsz * self.num_heads:
                    attn_mask = attn_mask.view(bsz, self.num_heads, tgt_len, src_len)
                else:
                    attn_mask = attn_mask.unsqueeze(1)
            if attn_mask.shape != attn.shape:
                attn_mask = attn_mask.expand_as(attn)
            attn = attn.masked_fill(attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else attn + attn_mask

        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = self.dropout_layer(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        out = self.out_proj(out)
        if not self.batch_first:
            out = out.transpose(0, 1)

        if need_weights:
            return out, attn.mean(dim=1) if average_attn_weights else attn
        return out, None


class LoRALayer(nn.Module):
    """The low-rank A/B pair: `x -> (x @ A @ B) * (alpha / rank)`."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)  # B starts at zero -> LoRA is a no-op initially

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    """A frozen `nn.Linear` plus a trainable `LoRALayer` added on top."""

    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original_layer
        for p in self.original_layer.parameters():
            p.requires_grad = False
        self.lora = LoRALayer(original_layer.in_features, original_layer.out_features, rank, alpha, dropout)

    @property
    def weight(self) -> torch.Tensor:
        return self.original_layer.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.original_layer.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_layer(x) + self.lora(x)


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: Optional[set] = None
    apply_to_vision_encoder: bool = True
    apply_to_text_encoder: bool = True
    apply_to_geometry_encoder: bool = True
    apply_to_detr_encoder: bool = True
    apply_to_detr_decoder: bool = True
    apply_to_mask_decoder: bool = True

    def __post_init__(self):
        self.target_modules = set(self.target_modules) if self.target_modules else set(DEFAULT_TARGET_MODULES)

    # module_name -> component flag, checked before the name-match in apply_lora_to_model
    _COMPONENT_MARKERS = (
        (("vision_encoder", "vision_backbone"), "apply_to_vision_encoder"),
        (("text_encoder", "language_backbone"), "apply_to_text_encoder"),
        (("geometry_encoder",), "apply_to_geometry_encoder"),
        (("detr_encoder", "transformer.encoder"), "apply_to_detr_encoder"),
        (("detr_decoder", "transformer.decoder"), "apply_to_detr_decoder"),
        (("mask_decoder",), "apply_to_mask_decoder"),
    )

    def component_enabled(self, module_name: str) -> bool:
        for markers, flag in self._COMPONENT_MARKERS:
            if any(m in module_name for m in markers) and not getattr(self, flag):
                return False
        return True

    def targets(self, module_name: str) -> bool:
        if not self.component_enabled(module_name):
            return False
        basename = module_name.rsplit(".", 1)[-1]
        return basename in self.target_modules or any(t in basename for t in self.target_modules)


def apply_lora_to_model(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Freeze `model`, then attach LoRA adapters per `config`.

    Two passes: (1) replace every `nn.MultiheadAttention` in an enabled
    component with `MultiheadAttentionLoRA` so its Q/K/V/out_proj become
    ordinary Linears, (2) wrap every Linear whose name matches
    `config.targets(...)` in a `LoRALinear`.
    """
    for p in model.parameters():
        p.requires_grad = False

    def _get_parent(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
        *path, attr = dotted_name.split(".")
        parent = root
        for p in path:
            parent = getattr(parent, p)
        return parent, attr

    mha_targets = [
        (name, m)
        for name, m in model.named_modules()
        if isinstance(m, nn.MultiheadAttention) and config.component_enabled(name)
    ]
    for name, mha in mha_targets:
        parent, attr = _get_parent(model, name)
        new_mha = MultiheadAttentionLoRA(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            bias=mha.in_proj_bias is not None,
            batch_first=mha.batch_first,
            in_proj_weight=mha.in_proj_weight,
            in_proj_bias=mha.in_proj_bias,
            out_proj_weight=mha.out_proj.weight,
            out_proj_bias=mha.out_proj.bias,
        )
        for p in new_mha.parameters():
            p.requires_grad = False
        setattr(parent, attr, new_mha)
    print(f"[lora] replaced {len(mha_targets)} nn.MultiheadAttention -> MultiheadAttentionLoRA")

    lora_targets = [
        name for name, m in model.named_modules() if isinstance(m, nn.Linear) and config.targets(name)
    ]
    for name in lora_targets:
        parent, attr = _get_parent(model, name)
        setattr(parent, attr, LoRALinear(getattr(parent, attr), config.rank, config.alpha, config.dropout))
    print(f"[lora] applied LoRA to {len(lora_targets)} Linear layers")

    return model


def count_parameters(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100 * trainable / total if total else 0.0,
    }


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only the LoRA A/B matrices (a few MB), not the frozen base model."""
    state = {
        f"{name}.lora_A": m.lora_A
        for name, m in model.named_modules()
        if isinstance(m, LoRALayer)
    }
    state.update(
        {f"{name}.lora_B": m.lora_B for name, m in model.named_modules() if isinstance(m, LoRALayer)}
    )
    torch.save(state, path)
    print(f"[lora] saved {len(state) // 2} LoRA layers -> {path}")


def load_lora_weights(model: nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `missing`/`unexpected` are expected to be large here: `strict=False` is
    # required because `state` only has LoRA params, not the frozen base model.
    print(f"[lora] loaded LoRA weights from {path} ({len(state)} tensors)")