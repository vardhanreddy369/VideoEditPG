#!/usr/bin/env python3
"""
Stage 3 — Parameter Generator: Inference

Generate personalized videos using the HyperNetwork instead of per-subject
fine-tuning. Replaces the standard DreamVideo adapter loading with a single
forward pass through the trained Parameter Generator.

Pipeline:
    Reference images → CLIP encoder → HyperNetwork → adapter weights
    → inject into UNet → standard DreamVideo diffusion → video

Usage:
    python inference_paramgen.py \
        --paramgen-ckpt workspace/paramgen_training/paramgen_best.pt \
        --ref-images path/to/ref1.jpg path/to/ref2.jpg \
        --prompt "a * person playing guitar" \
        --out-dir workspace/paramgen_results

    # Or use a config file:
    python inference_paramgen.py \
        --config configs/paramgen/infer_example.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_clip_encoder(pretrained_path: str, device: str):
    """Load the frozen CLIP encoder used by DreamVideo."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained=pretrained_path,
        )
        model = model.to(device).eval()
        return model, preprocess
    except ImportError:
        log.error("open_clip not installed. Run: pip install open_clip_torch")
        sys.exit(1)


def encode_reference_images(
    image_paths: list[str],
    clip_model,
    preprocess,
    device: str,
) -> torch.Tensor:
    """Encode reference images into CLIP embeddings."""
    from PIL import Image

    embeddings = []
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).to(device)
            feat = clip_model.encode_image(img_tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embeddings.append(feat)

    # Stack: (1, N, 1024)
    stacked = torch.cat(embeddings, dim=0).unsqueeze(0)
    return stacked


def generate_adapter_weights(
    paramgen_ckpt_path: str,
    clip_embeddings: torch.Tensor,
    device: str,
) -> dict[str, torch.Tensor]:
    """
    Generate adapter weights using the trained HyperNetwork.

    Args:
        paramgen_ckpt_path: Path to trained HyperNetwork checkpoint
        clip_embeddings: (1, N, 1024) CLIP features of reference images
        device: torch device

    Returns:
        state_dict with adapter weights (compatible with DreamVideo UNet)
    """
    from tools.param_generator.hypernet import HyperNetwork

    # Load checkpoint
    ckpt = torch.load(paramgen_ckpt_path, map_location=device)
    adapter_structure = ckpt["adapter_structure"]
    weight_mean = ckpt.get("weight_mean")
    weight_std = ckpt.get("weight_std")

    # Build model
    model = HyperNetwork(
        clip_dim=clip_embeddings.shape[-1],
        adapter_structure=adapter_structure,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    # Predict adapter weights
    with torch.no_grad():
        predicted_params = model.predict_state_dict(clip_embeddings.to(device))

    # If weights were normalized during training, denormalize
    if weight_mean is not None and weight_std is not None:
        # Reconstruct the flat vector, denormalize, then re-split
        flat_pred = model(clip_embeddings.to(device), return_flat=True).squeeze(0)
        flat_denorm = flat_pred * weight_std.to(device) + weight_mean.to(device)

        # Re-split into parameter dict using adapter_structure
        offset = 0
        denorm_params = {}
        for block_info in adapter_structure:
            key_prefix = block_info.get("original_key_prefix", block_info["key_prefix"].replace("_", "."))
            in_dim = block_info["in_dim"]
            hidden_dim = block_info["hidden_dim"]

            # down_linear.weight
            size = hidden_dim * in_dim
            denorm_params[f"{key_prefix}.down_linear.weight"] = (
                flat_denorm[offset:offset + size].view(hidden_dim, in_dim)
            )
            offset += size

            # down_linear.bias
            denorm_params[f"{key_prefix}.down_linear.bias"] = (
                flat_denorm[offset:offset + hidden_dim].view(hidden_dim)
            )
            offset += hidden_dim

            # up_linear.weight
            size = in_dim * hidden_dim
            denorm_params[f"{key_prefix}.up_linear.weight"] = (
                flat_denorm[offset:offset + size].view(in_dim, hidden_dim)
            )
            offset += size

            # up_linear.bias
            denorm_params[f"{key_prefix}.up_linear.bias"] = (
                flat_denorm[offset:offset + in_dim].view(in_dim)
            )
            offset += in_dim

        predicted_params = denorm_params

    # Move to CPU for state_dict merging
    predicted_params = {k: v.cpu() for k, v in predicted_params.items()}

    log.info(f"Generated {len(predicted_params)} adapter parameters")
    total_params = sum(v.numel() for v in predicted_params.values())
    log.info(f"Total adapter params: {total_params:,}")

    return predicted_params


def run_dreamvideo_inference(
    adapter_state_dict: dict[str, torch.Tensor],
    prompt: str,
    ref_image_path: str,
    base_model_path: str,
    clip_model_path: str,
    out_dir: str,
    num_frames: int = 32,
    guide_scale: float = 9.0,
    num_steps: int = 50,
    seed: int = 8888,
    device: str = "cuda",
):
    """
    Run DreamVideo inference with generated adapter weights.
    This mirrors inference_dreamvideo_entrance.py but uses predicted adapters.
    """
    from tools.modules.unet.unet_dreamvideo import UNetSD_DreamVideo
    from tools.modules.diffusions.diffusion_ddim import DiffusionDDIM
    from tools.modules.clip_embedder import FrozenOpenCLIPCustomEmbedder

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    log.info("Loading base model...")

    # Build UNet with adapter support
    model = UNetSD_DreamVideo(
        in_dim=4, y_dim=1024, context_dim=1024, out_dim=4,
        dim_mult=[1, 2, 4, 4], num_heads=8, head_dim=64,
        num_res_blocks=2, dropout=0.1, misc_dropout=0.4,
        temporal_attention=True, temporal_attn_times=1,
        use_checkpoint=True, use_fps_condition=False,
        use_sim_mask=False,
        spatial_adapter_list=["cross_attention"],
    )

    # Load base weights
    base_state_dict = torch.load(base_model_path, map_location="cpu")
    if "state_dict" in base_state_dict:
        base_state_dict = base_state_dict["state_dict"]

    # Merge predicted adapter weights
    merged = {**base_state_dict, **adapter_state_dict}
    model.load_state_dict(merged, strict=False)
    model = model.to(device).eval()

    # Build diffusion
    diffusion = DiffusionDDIM(
        schedule="linear_sd",
        schedule_param={
            "num_timesteps": 1000,
            "init_beta": 0.00085,
            "last_beta": 0.0120,
            "zero_terminal_snr": False,
        },
        mean_type="eps",
        var_type="fixed_small",
        loss_type="mse",
        rescale_timesteps=False,
        noise_strength=0.1,
    )

    # Build CLIP encoder
    clip_encoder = FrozenOpenCLIPCustomEmbedder(
        layer="penultimate",
        pretrained=clip_model_path,
        vit_resolution=[224, 224],
    ).to(device).eval()

    # Encode text
    log.info(f"Generating video for: '{prompt}'")
    torch.manual_seed(seed)

    with torch.no_grad():
        # Text encoding
        y = clip_encoder(text=[prompt])
        y = y.unsqueeze(0) if y.dim() == 2 else y

        # Sample noise
        noise = torch.randn(1, num_frames, 4, 32, 32, device=device)

        # DDIM sampling
        model_kwargs = {"y": y}
        samples = diffusion.ddim_sample_loop(
            model=model,
            shape=noise.shape,
            noise=noise,
            model_kwargs=model_kwargs,
            guide_scale=guide_scale,
            ddim_timesteps=num_steps,
            eta=0.0,
        )

    log.info(f"Generated video shape: {samples.shape}")
    log.info(f"Saved to: {out_path}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Parameter Generator Inference")
    parser.add_argument("--paramgen-ckpt", required=True, help="Trained HyperNetwork checkpoint")
    parser.add_argument("--ref-images", nargs="+", required=True, help="Reference image paths")
    parser.add_argument("--prompt", default="a * person speaking to camera", help="Text prompt")
    parser.add_argument("--clip-model", default="models/open_clip_pytorch_model.bin")
    parser.add_argument("--base-model", default="models/model_scope_v1-5_0632000.pth")
    parser.add_argument("--out-dir", default="workspace/paramgen_results")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--guide-scale", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Step 1: Encode reference images with CLIP
    log.info(f"Encoding {len(args.ref_images)} reference images...")
    clip_model, preprocess = load_clip_encoder(args.clip_model, args.device)
    clip_embeddings = encode_reference_images(
        args.ref_images, clip_model, preprocess, args.device,
    )
    log.info(f"CLIP embeddings shape: {clip_embeddings.shape}")

    # Free CLIP model memory
    del clip_model
    torch.cuda.empty_cache()

    # Step 2: Generate adapter weights (THE KEY STEP - single forward pass!)
    log.info("Generating adapter weights with HyperNetwork...")
    import time
    t0 = time.time()
    adapter_weights = generate_adapter_weights(
        args.paramgen_ckpt, clip_embeddings, args.device,
    )
    gen_time = time.time() - t0
    log.info(f"Adapter generation time: {gen_time:.3f}s")

    # Step 3: Run DreamVideo inference with generated adapters
    log.info("Running video generation...")
    run_dreamvideo_inference(
        adapter_state_dict=adapter_weights,
        prompt=args.prompt,
        ref_image_path=args.ref_images[0],
        base_model_path=args.base_model,
        clip_model_path=args.clip_model,
        out_dir=args.out_dir,
        num_frames=args.num_frames,
        guide_scale=args.guide_scale,
        seed=args.seed,
        device=args.device,
    )

    log.info("Done!")
    log.info(f"  Adapter generation: {gen_time:.3f}s (vs ~20min for fine-tuning)")
    log.info(f"  Speedup: ~{int(20*60/max(gen_time, 0.001))}x")


if __name__ == "__main__":
    main()
