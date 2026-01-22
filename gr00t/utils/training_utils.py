from __future__ import annotations

import torch
from torch import nn

from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear


def init_weights_vit_timm(module: nn.Module, name: str = ""):
    """ViT-style weight initialization (timm-like) for reproducibility/stability.

    Notes:
    - We intentionally support `CategorySpecificLinear` (multi-embodiment weights).
    - The `name` argument is kept for compatibility with `named_apply`.
    """
    _ = name

    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        return

    if isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)
        return

    if isinstance(module, CategorySpecificLinear):
        nn.init.trunc_normal_(module.W, std=0.02)
        if module.b is not None:
            nn.init.zeros_(module.b)
        return


def named_apply(
    fn,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Apply `fn(module=..., name=...)` recursively to a module tree.

    This mirrors the common pattern used in ViT/timm init utilities and is handy
    when adding new submodules (e.g., cross-attn blocks) that require explicit init.
    """
    if not depth_first and include_root:
        fn(module=module, name=name)

    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )

    if depth_first and include_root:
        fn(module=module, name=name)

    return module