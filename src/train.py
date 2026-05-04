#!/usr/bin/env python3
"""
Stage 3 — Parameter Generator: Training Loop

Trains the HyperNetwork to predict DreamVideo identity adapter weights
from CLIP embeddings.

Training data: (clip_embedding, ground_truth_adapter_weights) pairs
produced by the LoRA Factory (Stage 2).

Loss functions:
    L_weight: MSE between predicted and ground-truth adapter weights
    L_cosine: Cosine similarity loss on flattened weight vectors
    L_reg:    L2 regularization on predicted weights (keep small)

Usage:
    python train_paramgen.py \
        --dataset workspace/paramgen_dataset/paramgen_dataset.pt \
        --out-dir workspace/paramgen_training \
        --epochs 500 \
        --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tools.param_generator.hypernet import HyperNetwork

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class AdapterDataset(Dataset):
    """Dataset of (CLIP embedding, adapter weights) pairs."""

    def __init__(self, dataset_path: str, normalize_weights: bool = True):
        data = torch.load(dataset_path, map_location="cpu")

        self.clip_embeddings = data["clip_embeddings"]  # (N, 1024)
        self.adapter_weights = data["adapter_weights"]  # (N, total_params)
        self.adapter_structure = data["adapter_structure"]

        self.normalize_weights = normalize_weights
        if normalize_weights:
            self.weight_mean = data["adapter_mean"]
            self.weight_std = data["adapter_std"]
            # Normalize targets to zero-mean, unit-variance
            self.adapter_weights_norm = (self.adapter_weights - self.weight_mean) / self.weight_std
        else:
            self.weight_mean = None
            self.weight_std = None
            self.adapter_weights_norm = self.adapter_weights

        log.info(f"Loaded dataset: {len(self)} samples")
        log.info(f"  CLIP dim: {self.clip_embeddings.shape[-1]}")
        log.info(f"  Adapter params: {self.adapter_weights.shape[-1]:,}")

    def __len__(self) -> int:
        return len(self.clip_embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.clip_embeddings[idx],
            self.adapter_weights_norm[idx],
            self.adapter_weights[idx],  # unnormalized for evaluation
        )


class ParamGenTrainer:
    """Training loop for the Parameter Generator."""

    def __init__(
        self,
        model: HyperNetwork,
        dataset: AdapterDataset,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 8,
        epochs: int = 500,
        warmup_steps: int = 100,
        loss_weights: dict | None = None,
        out_dir: str = "workspace/paramgen_training",
        device: str = "cuda",
        val_split: float = 0.15,
    ):
        self.device = device
        self.epochs = epochs
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.model = model.to(device)
        self.dataset = dataset

        # Loss weights
        self.loss_weights = loss_weights or {
            "weight_mse": 1.0,
            "cosine": 0.5,
            "l2_reg": 0.01,
        }

        # Train/val split
        n_val = max(1, int(len(dataset) * val_split))
        n_train = len(dataset) - n_val
        self.train_set, self.val_set = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_loader = DataLoader(
            self.train_set, batch_size=min(batch_size, n_train),
            shuffle=True, drop_last=False, num_workers=0,
        )
        self.val_loader = DataLoader(
            self.val_set, batch_size=min(batch_size, n_val),
            shuffle=False, drop_last=False, num_workers=0,
        )

        log.info(f"Train: {n_train}, Val: {n_val}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        # Cosine LR scheduler with warmup
        total_steps = epochs * len(self.train_loader)
        self.warmup_steps = warmup_steps

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        import math
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.global_step = 0
        self.best_val_loss = float("inf")

    def compute_loss(
        self,
        pred_flat: torch.Tensor,
        target_flat: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss."""
        losses = {}

        # L1: MSE on normalized weights
        loss_mse = F.mse_loss(pred_flat, target_flat)
        losses["weight_mse"] = loss_mse.item()

        # L2: Cosine similarity loss (1 - cos_sim)
        cos_sim = F.cosine_similarity(
            pred_flat.flatten(1), target_flat.flatten(1), dim=1
        ).mean()
        loss_cosine = 1.0 - cos_sim
        losses["cosine"] = loss_cosine.item()
        losses["cos_sim"] = cos_sim.item()

        # L3: L2 regularization (keep predicted weights small)
        loss_reg = pred_flat.pow(2).mean()
        losses["l2_reg"] = loss_reg.item()

        # Combined
        total = (
            self.loss_weights["weight_mse"] * loss_mse
            + self.loss_weights["cosine"] * loss_cosine
            + self.loss_weights["l2_reg"] * loss_reg
        )
        losses["total"] = total.item()

        return total, losses

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on validation set."""
        self.model.eval()
        total_losses = {}
        n_batches = 0

        for clip_emb, target_norm, target_raw in self.val_loader:
            clip_emb = clip_emb.to(self.device)
            target_norm = target_norm.to(self.device)

            pred_flat = self.model(clip_emb, return_flat=True)
            _, losses = self.compute_loss(pred_flat, target_norm)

            for k, v in losses.items():
                total_losses[k] = total_losses.get(k, 0) + v
            n_batches += 1

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        self.model.train()
        return avg_losses

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint."""
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "adapter_structure": self.dataset.adapter_structure,
            "weight_mean": self.dataset.weight_mean,
            "weight_std": self.dataset.weight_std,
        }

        # Save periodic checkpoint
        ckpt_path = self.out_dir / f"paramgen_epoch_{epoch:04d}.pt"
        torch.save(ckpt, ckpt_path)

        # Save best
        if is_best:
            best_path = self.out_dir / "paramgen_best.pt"
            torch.save(ckpt, best_path)
            log.info(f"  New best model saved: val_loss={val_loss:.6f}")

    def train(self):
        """Main training loop."""
        log.info(f"Starting training for {self.epochs} epochs")
        log.info(f"  Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        log.info(f"  Loss weights: {self.loss_weights}")

        history = []

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_losses = {}
            n_batches = 0
            t0 = time.time()

            for clip_emb, target_norm, target_raw in self.train_loader:
                clip_emb = clip_emb.to(self.device)
                target_norm = target_norm.to(self.device)

                # Forward
                pred_flat = self.model(clip_emb, return_flat=True)
                loss, losses = self.compute_loss(pred_flat, target_norm)

                # Backward
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()

                for k, v in losses.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v
                n_batches += 1
                self.global_step += 1

            # Average epoch losses
            avg_train = {k: v / n_batches for k, v in epoch_losses.items()}
            dt = time.time() - t0

            # Evaluate
            avg_val = self.evaluate()

            # Log
            entry = {
                "epoch": epoch,
                "lr": self.scheduler.get_last_lr()[0],
                "train": avg_train,
                "val": avg_val,
                "time_s": dt,
            }
            history.append(entry)

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    f"Epoch {epoch:4d}/{self.epochs} | "
                    f"train_loss={avg_train['total']:.6f} | "
                    f"val_loss={avg_val['total']:.6f} | "
                    f"cos_sim={avg_val.get('cos_sim', 0):.4f} | "
                    f"lr={entry['lr']:.2e} | "
                    f"{dt:.1f}s"
                )

            # Save checkpoints
            is_best = avg_val["total"] < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val["total"]

            if epoch % 50 == 0 or is_best:
                self.save_checkpoint(epoch, avg_val["total"], is_best)

        # Save final
        self.save_checkpoint(self.epochs, avg_val["total"])

        # Save training history
        with open(self.out_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        log.info(f"Training complete. Best val loss: {self.best_val_loss:.6f}")
        return history


def main():
    parser = argparse.ArgumentParser(description="Train Parameter Generator")
    parser.add_argument("--dataset", required=True, help="Path to paramgen_dataset.pt")
    parser.add_argument("--out-dir", default="workspace/paramgen_training")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--backbone-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--from-checkpoint", default="", help="Path to adapter checkpoint for auto structure detection")
    parser.add_argument("--loss-weight-mse", type=float, default=1.0)
    parser.add_argument("--loss-weight-cosine", type=float, default=0.5)
    parser.add_argument("--loss-weight-reg", type=float, default=0.01)
    args = parser.parse_args()

    # Load dataset
    dataset = AdapterDataset(args.dataset, normalize_weights=True)

    # Create model
    if args.from_checkpoint:
        model = HyperNetwork.from_adapter_checkpoint(
            args.from_checkpoint,
            hidden_dim=args.hidden_dim,
            num_backbone_layers=args.backbone_layers,
            dropout=args.dropout,
        )
    else:
        model = HyperNetwork(
            clip_dim=dataset.clip_embeddings.shape[-1],
            hidden_dim=args.hidden_dim,
            num_backbone_layers=args.backbone_layers,
            dropout=args.dropout,
        )

    # Train
    trainer = ParamGenTrainer(
        model=model,
        dataset=dataset,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        loss_weights={
            "weight_mse": args.loss_weight_mse,
            "cosine": args.loss_weight_cosine,
            "l2_reg": args.loss_weight_reg,
        },
        out_dir=args.out_dir,
        device=args.device,
        val_split=args.val_split,
    )

    trainer.train()


if __name__ == "__main__":
    main()
