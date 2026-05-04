"""
Stage 3 — Parameter Generator: HyperNetwork Architecture

Predicts DreamVideo identity adapter weights from CLIP image embeddings.
Given reference images of a subject, generates the full set of adapter
parameters in a single forward pass — no per-subject fine-tuning needed.

Architecture:
    Reference Images (N × 3 × H × W)
        ↓
    CLIP ViT-H Encoder (frozen) → (N, 1024)
        ↓
    Attention Pooling across N images → (1024,)
        ↓
    Shared MLP Backbone → (hidden_dim,)
        ↓
    Per-Block Weight Heads → adapter state_dict
        ↓
    Inject into frozen DreamVideo UNet → Personalized Video

The adapter structure matches DreamVideo exactly:
    Each adapter block = Adapter(in_dim=D, hidden_dim=D//2)
        - down_linear.weight: (D//2, D)
        - down_linear.bias:   (D//2,)
        - up_linear.weight:   (D, D//2)
        - up_linear.bias:     (D,)
    Total per block: D*D//2 + D//2 + D*D//2 + D = D² + 3D//2
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """Pool N image embeddings into a single embedding using learned attention."""

    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) — N image embeddings per sample
        Returns:
            (B, D) — pooled embedding
        """
        B = x.shape[0]
        query = self.query.expand(B, -1, -1)  # (B, 1, D)
        out, _ = self.attn(query, x, x)  # (B, 1, D)
        out = self.norm(out.squeeze(1))  # (B, D)
        return out


class WeightHead(nn.Module):
    """Predicts weights for a single adapter block."""

    def __init__(self, input_dim: int, adapter_in_dim: int, adapter_hidden_dim: int):
        super().__init__()
        self.adapter_in_dim = adapter_in_dim
        self.adapter_hidden_dim = adapter_hidden_dim

        # Parameter counts for this adapter
        self.down_w_size = adapter_hidden_dim * adapter_in_dim
        self.down_b_size = adapter_hidden_dim
        self.up_w_size = adapter_in_dim * adapter_hidden_dim
        self.up_b_size = adapter_in_dim
        self.total_params = (
            self.down_w_size + self.down_b_size +
            self.up_w_size + self.up_b_size
        )

        # Two-layer MLP to predict adapter weights
        mid_dim = min(input_dim, self.total_params // 2)
        mid_dim = max(mid_dim, 256)  # ensure minimum capacity

        self.head = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.GELU(),
            nn.LayerNorm(mid_dim),
            nn.Linear(mid_dim, self.total_params),
        )

        # Initialize output layer to near-zero so initial adapters ≈ identity
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) — subject embedding
        Returns:
            dict with keys: down_linear.weight, down_linear.bias,
                           up_linear.weight, up_linear.bias
        """
        B = x.shape[0]
        raw = self.head(x)  # (B, total_params)

        # Split into individual parameter tensors
        offset = 0

        down_w = raw[:, offset:offset + self.down_w_size]
        down_w = down_w.view(B, self.adapter_hidden_dim, self.adapter_in_dim)
        offset += self.down_w_size

        down_b = raw[:, offset:offset + self.down_b_size]
        down_b = down_b.view(B, self.adapter_hidden_dim)
        offset += self.down_b_size

        up_w = raw[:, offset:offset + self.up_w_size]
        up_w = up_w.view(B, self.adapter_in_dim, self.adapter_hidden_dim)
        offset += self.up_w_size

        up_b = raw[:, offset:offset + self.up_b_size]
        up_b = up_b.view(B, self.adapter_in_dim)

        return {
            "down_linear.weight": down_w,
            "down_linear.bias": down_b,
            "up_linear.weight": up_w,
            "up_linear.bias": up_b,
        }


class HyperNetwork(nn.Module):
    """
    HyperNetwork that generates DreamVideo identity adapter parameters
    from CLIP image embeddings.

    Given CLIP features of reference images, predicts all adapter weights
    in a single forward pass.
    """

    def __init__(
        self,
        clip_dim: int = 1024,
        hidden_dim: int = 2048,
        num_backbone_layers: int = 4,
        adapter_structure: Optional[list[dict]] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            clip_dim: CLIP embedding dimension (1024 for ViT-H-14)
            hidden_dim: Hidden dimension of the shared backbone
            num_backbone_layers: Number of MLP layers in the backbone
            adapter_structure: List of dicts describing each adapter block.
                Each dict has: {"key_prefix": str, "in_dim": int, "hidden_dim": int}
                If None, uses default DreamVideo identity adapter structure.
            dropout: Dropout rate
        """
        super().__init__()
        self.clip_dim = clip_dim
        self.hidden_dim = hidden_dim

        # Attention pooling for multiple reference images
        self.attention_pool = AttentionPooling(clip_dim)

        # Shared backbone
        layers = []
        in_dim = clip_dim
        for i in range(num_backbone_layers):
            out_dim = hidden_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        self.backbone = nn.Sequential(*layers)

        # Set up adapter structure
        if adapter_structure is None:
            adapter_structure = self._default_identity_adapter_structure()
        self.adapter_structure = adapter_structure

        # Per-block weight heads
        self.weight_heads = nn.ModuleDict()
        self.total_adapter_params = 0

        for block_info in adapter_structure:
            key = block_info["key_prefix"].replace(".", "_")
            head = WeightHead(
                input_dim=hidden_dim,
                adapter_in_dim=block_info["in_dim"],
                adapter_hidden_dim=block_info["hidden_dim"],
            )
            self.weight_heads[key] = head
            self.total_adapter_params += head.total_params

        print(f"HyperNetwork initialized:")
        print(f"  Adapter blocks: {len(adapter_structure)}")
        print(f"  Total adapter params to predict: {self.total_adapter_params:,}")
        print(f"  HyperNetwork own params: {sum(p.numel() for p in self.parameters()):,}")

    @staticmethod
    def _default_identity_adapter_structure() -> list[dict]:
        """
        Default DreamVideo identity adapter structure.
        Identity adapters are cross_attn_adapter in SpatialTransformerWithAdapter blocks.

        The UNet has adapters at these locations (context_dim=1024, hidden=512):
            input_blocks: blocks 1,2 (dim=320), 4,5 (dim=640), 7,8 (dim=1280)
            middle_block: block 1 (dim=1280)
            output_blocks: blocks 3,4,5 (dim=1280), 6,7,8 (dim=640), 9,10,11 (dim=320)

        Each SpatialTransformer has depth=1 transformer block with cross_attn_adapter.
        Adapter in_dim = inner_dim = n_heads * d_head (varies by block).
        """
        # These are the spatial transformer dimensions at each UNet level
        # dim_mult = [1, 2, 4, 4], base_dim = 320
        # Level 0: 320, Level 1: 640, Level 2: 1280, Level 3: 1280
        structure = []

        # Input blocks with spatial transformers (indices approximate)
        dims = [
            ("input_blocks_1_1_transformer_blocks_0", 320),
            ("input_blocks_2_1_transformer_blocks_0", 320),
            ("input_blocks_4_1_transformer_blocks_0", 640),
            ("input_blocks_5_1_transformer_blocks_0", 640),
            ("input_blocks_7_1_transformer_blocks_0", 1280),
            ("input_blocks_8_1_transformer_blocks_0", 1280),
            # Middle block
            ("middle_block_1_transformer_blocks_0", 1280),
            # Output blocks
            ("output_blocks_3_1_transformer_blocks_0", 1280),
            ("output_blocks_4_1_transformer_blocks_0", 1280),
            ("output_blocks_5_1_transformer_blocks_0", 1280),
            ("output_blocks_6_1_transformer_blocks_0", 640),
            ("output_blocks_7_1_transformer_blocks_0", 640),
            ("output_blocks_8_1_transformer_blocks_0", 640),
            ("output_blocks_9_1_transformer_blocks_0", 320),
            ("output_blocks_10_1_transformer_blocks_0", 320),
            ("output_blocks_11_1_transformer_blocks_0", 320),
        ]

        for key_prefix, dim in dims:
            structure.append({
                "key_prefix": key_prefix,
                "in_dim": dim,
                "hidden_dim": dim // 2,
            })

        return structure

    @classmethod
    def from_adapter_checkpoint(cls, checkpoint_path: str, clip_dim: int = 1024, **kwargs) -> "HyperNetwork":
        """
        Create a HyperNetwork whose output structure matches an existing
        adapter checkpoint, automatically detecting dimensions.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)

        # Parse adapter structure from checkpoint keys
        adapter_blocks = {}
        for key, tensor in state_dict.items():
            if "adapter" not in key:
                continue
            # Key format: input_blocks.1.1.transformer_blocks.0.cross_attn_adapter.down_linear.weight
            parts = key.split(".")
            # Find adapter name position
            for i, part in enumerate(parts):
                if "adapter" in part:
                    prefix = ".".join(parts[:i + 1])
                    param_name = ".".join(parts[i + 1:])
                    if prefix not in adapter_blocks:
                        adapter_blocks[prefix] = {}
                    adapter_blocks[prefix][param_name] = tensor
                    break

        structure = []
        for prefix, params in sorted(adapter_blocks.items()):
            down_w = params.get("down_linear.weight")
            if down_w is None:
                continue
            hidden_dim, in_dim = down_w.shape
            safe_key = prefix.replace(".", "_")
            structure.append({
                "key_prefix": safe_key,
                "original_key_prefix": prefix,
                "in_dim": in_dim,
                "hidden_dim": hidden_dim,
            })

        print(f"Detected {len(structure)} adapter blocks from checkpoint")
        for s in structure:
            print(f"  {s['original_key_prefix']}: in_dim={s['in_dim']}, hidden_dim={s['hidden_dim']}")

        return cls(clip_dim=clip_dim, adapter_structure=structure, **kwargs)

    def forward(
        self,
        clip_embeddings: torch.Tensor,
        return_flat: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Generate adapter parameters from CLIP embeddings.

        Args:
            clip_embeddings: (B, N, D) for N reference images, or (B, D) for pre-pooled
            return_flat: If True, return flattened parameter vector instead of dict

        Returns:
            If return_flat=False: dict mapping adapter parameter names to tensors
            If return_flat=True: (B, total_params) flattened tensor
        """
        # Handle different input shapes
        if clip_embeddings.dim() == 2:
            # Already pooled: (B, D)
            pooled = clip_embeddings
        elif clip_embeddings.dim() == 3:
            # Multiple refs: (B, N, D) → pool to (B, D)
            pooled = self.attention_pool(clip_embeddings)
        else:
            raise ValueError(f"Expected 2D or 3D input, got {clip_embeddings.dim()}D")

        # Shared backbone
        features = self.backbone(pooled)  # (B, hidden_dim)

        # Generate weights for each adapter block
        if return_flat:
            flat_parts = []
            for block_info in self.adapter_structure:
                key = block_info["key_prefix"].replace(".", "_")
                block_params = self.weight_heads[key](features)
                for param_name in ["down_linear.weight", "down_linear.bias",
                                   "up_linear.weight", "up_linear.bias"]:
                    flat_parts.append(block_params[param_name].flatten(1))
            return torch.cat(flat_parts, dim=1)

        # Return as state_dict-like structure
        all_params = {}
        for block_info in self.adapter_structure:
            key = block_info["key_prefix"].replace(".", "_")
            original_key = block_info.get("original_key_prefix", key.replace("_", "."))
            block_params = self.weight_heads[key](features)
            for param_name, tensor in block_params.items():
                full_key = f"{original_key}.{param_name}"
                all_params[full_key] = tensor

        return all_params

    def predict_state_dict(self, clip_embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Generate a state_dict compatible with model.load_state_dict() for a single sample.

        Args:
            clip_embeddings: (1, N, D) or (1, D) — single subject

        Returns:
            state_dict with adapter parameters (no batch dimension)
        """
        params = self.forward(clip_embeddings)
        return {k: v.squeeze(0) for k, v in params.items()}
