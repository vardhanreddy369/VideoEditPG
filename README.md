# VideoEditPG

**Parameter Generation for Human-Driven Video Editing**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

> CAP6614: Efficient AI — Final Project | University of Central Florida, Spring 2026

---

## The Problem

Personalizing a video diffusion model to a specific human identity currently requires **per-subject fine-tuning**: minutes of compute and gigabytes of storage per person. This does not scale.

## Our Approach

We train a **hypernetwork once** — it learns to predict personalized LoRA adapter weights from reference images in a single forward pass, with no gradient descent at inference time.

```text
Reference Images  ──►  CLIP ViT-H  ──►  HyperNetwork  ──►  LoRA Weights
                                                                   │
                                                                   ▼
                                          Frozen Wan2.1-VACE  +  ΔW  ──►  Edited Video
```

---

## Results

All experiments run on Newton HPC — Tesla V100-PCIE-32GB.

### Identity Generation (Wan2.1-VACE-1.3B)

| Subject | Prompt | Time (s) | VRAM (GB) |
| --- | --- | --- | --- |
| woman_blonde_smile | Young blonde woman waves at camera | 66.1 | 21.06 |
| man_beard_walk | Man with dark beard walks through city | 49.4 | 21.06 |
| woman_curly_dance | Woman with curly hair dances in studio | 49.6 | 21.06 |
| man_asian_talk | Young man talks in coffee shop | 49.5 | 21.06 |
| woman_elderly_garden | Elderly woman tends flowers in garden | 49.4 | 21.06 |

### LoRA Injection Feasibility

| Metric | Value |
| --- | --- |
| LoRA rank | 4 |
| LoRA parameters | 614,400 (0.04% of model) |
| Baseline inference time | 33.8s |
| With LoRA inference time | 35.0s (+3.6%) |
| MSE vs baseline | 0.05 |
| Target modules | 50 |

LoRA can be injected into Wan2.1-VACE with **negligible overhead** and no quality degradation — confirming the backbone is compatible with our approach.

---

## Architecture

### HyperNetwork

```text
Reference Images (N × 3 × H × W)
        │
        ▼
  CLIP ViT-H Encoder  ──  frozen, no gradient
  (N embeddings × 1024-dim)
        │
        ▼
  Attention Pooling  ──  learned query pooling across N refs
  (1024-dim)
        │
        ▼
  Shared MLP Backbone  ──  4 layers, 2048-dim hidden
  (2048-dim)
        │
        ▼
  Per-Layer Weight Heads  ──  one head per UNet adapter block
        │
        ▼
  LightLoRA Weights: {A_pred, B_pred} per layer
```

### LightLoRA Decomposition

Standard LoRA: `ΔW = A × B`

**LightLoRA (ours):**

```text
ΔW = A_aux  ×  A_pred  ×  B_pred  ×  B_aux
      ↑              ↑        ↑              ↑
   frozen         predicted predicted      frozen
  shared          per-subject              shared
```

- `A_aux`, `B_aux` — frozen orthogonal bases, shared across all identities
- `A_pred`, `B_pred` — subject-specific matrices, predicted by HyperNetwork
- Rank = 4 | ~614K parameters | <50KB per identity

### Base Model

**Wan2.1-VACE-1.3B** — chosen over HunyuanVideo and CogVideoX-5B for:

| Criterion | Wan2.1-VACE | HunyuanVideo | CogVideoX-5B |
| --- | --- | --- | --- |
| Native video editing | ✅ VACE | ❌ None | Partial |
| LoRA ecosystem | ✅ Mature | Minimal | Limited |
| 1.3B variant for prototyping | ✅ | ❌ (8.3B min) | ❌ |
| Open editing models built on it | 3 (VACE, Kiwi-Edit, SAMA) | 0 | 1 |
| VRAM for 1.3B | 8GB | — | — |

---

## Setup

### 1. Clone

```bash
git clone https://github.com/vardhanreddy369/VideoEditPG.git
cd VideoEditPG
```

### 2. Environment

**Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate videoeditpg
```

**Pip:**

```bash
pip install -r requirements.txt
```

### 3. Download Base Model

```bash
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --local-dir ./models/wan_vace_1_3b
```

---

## Reproducing Results

### Identity Generation

```bash
python src/inference.py \
  --mode identity \
  --prompt "A young woman with curly hair dances in a studio, white background" \
  --model-dir ./models/wan_vace_1_3b \
  --output-dir ./results/identity_generation
```

### LoRA Injection Test

Verifies that LoRA weights can be injected into Wan2.1-VACE with no overhead:

```bash
python src/inference.py \
  --mode lora_test \
  --lora-rank 4 \
  --model-dir ./models/wan_vace_1_3b \
  --output-dir ./results/lora_test
```

Expected output:

```text
lora_rank:       4
lora_params:     614,400
lora_percent:    0.04%
baseline_time:   33.8s
lora_time:       35.0s
mse:             0.05
```

### Train the HyperNetwork

```bash
python src/train.py \
  --dataset ./data/paramgen_dataset.pt \
  --out-dir ./checkpoints \
  --epochs 500 \
  --lr 1e-4 \
  --batch-size 8
```

**On Newton HPC (SLURM):**

```bash
sbatch scripts/newton_train.sh
```

### Run Full Pipeline (after training)

```bash
python src/inference.py \
  --paramgen-ckpt ./checkpoints/paramgen_best.pt \
  --ref-images ./data/ref1.jpg ./data/ref2.jpg \
  --prompt "a person speaking on stage" \
  --output-dir ./results/personalized
```

---

## Repository Structure

```text
VideoEditPG/
├── src/
│   ├── hypernet.py       # HyperNetwork + LightLoRA architecture
│   ├── train.py          # Training loop with cosine LR + weight-space loss
│   └── inference.py      # Full inference pipeline
├── scripts/
│   └── newton_train.sh   # SLURM job script for Newton HPC
├── configs/
│   └── train.yaml        # Training hyperparameters
├── results/
│   └── test_results.json # Experiment results from Newton
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Training Details

| Hyperparameter | Value |
| --- | --- |
| CLIP encoder | ViT-H-14 (frozen) |
| Hidden dim | 2048 |
| Backbone layers | 4 |
| Loss | MSE + Cosine similarity + L2 reg |
| Optimizer | AdamW (lr=1e-4, wd=0.01) |
| LR schedule | Cosine with 100-step warmup |
| Epochs | 500 |
| Batch size | 8 |
| Gradient clipping | 1.0 |

**Loss function:**

```text
L = λ₁ · MSE(ΔW_pred, ΔW_gt)  +  λ₂ · (1 − cosine_sim)  +  λ₃ · ||ΔW_pred||²
    λ₁=1.0                         λ₂=0.5                     λ₃=0.01
```

The L2 weight-space loss is the key insight — it supervises in weight space, not just pixel space, giving the hypernetwork a direct target.

---

## Limitations

- VACE text-guided editing pipeline requires `ftfy` — dependency fix in progress
- Hypernetwork currently trained on face identities; generalization to arbitrary subjects is future work
- Full end-to-end training pending compute allocation — current results demonstrate LoRA injection feasibility on the target backbone

---

## Related Work

| Paper | Venue | Relevance |
| --- | --- | --- |
| [HyperDreamBooth](https://arxiv.org/abs/2307.06949) | CVPR 2023 | Hypernetwork for image personalization — our primary inspiration |
| [Video2LoRA](https://arxiv.org/abs/2603.08210) | CVPR 2026 | Parameter generation for video generation |
| [VACE](https://github.com/ali-vilab/VACE) | 2025 | Our editing backbone |
| [HY-WU](https://arxiv.org/abs/2603.xxxxx) | 2026 | Parameter generation for image editing (Tencent) |

---

## Team

| Name | Contribution |
| --- | --- |
| Sri Vardhan Reddy Gutta | Architecture design, LightLoRA implementation, Newton experiments, LoRA injection |
| Bhargavi | Dataset curation, VACE pipeline integration |
| Samhith Reddy | Training pipeline, SLURM scripts, evaluation framework |
| Sarayu | Related work survey, analysis, report |

---

## License

MIT
