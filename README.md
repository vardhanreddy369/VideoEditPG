# VideoEditPG — Parameter Generation for Human-Driven Video Editing

**CAP6614: Efficient AI — Final Project**
**University of Central Florida, Spring 2026**

> A hypernetwork that predicts LoRA weights for personalized human video editing — combining Video2LoRA's lightweight LightLoRA decomposition with Wan2.1-VACE as the editing backbone.

---

## Overview

Existing video editing methods require per-subject fine-tuning, taking minutes of compute and gigabytes of storage per identity. **VideoEditPG** introduces a parameter generator — a hypernetwork trained once that predicts personalized LoRA weights from reference images in a single forward pass.

**Key idea**: Given reference images of a subject + a text editing instruction, the hypernetwork outputs LoRA weight deltas that are injected into a frozen Wan2.1-VACE backbone to perform identity-preserving video editing.

### Results at a glance

| | Baseline (no LoRA) | With Predicted LoRA |
|---|---|---|
| Identity generation | ✅ 5 videos, ~50s each | — |
| LoRA injection overhead | 33.8s | 35.0s (+3.6%) |
| LoRA parameters | — | 614K (0.04% of model) |
| VRAM (V100-32GB) | 21.06 GB | 21.06 GB |

---

## Team

| Name | Contribution |
|---|---|
| Sri Vardhan Reddy Gutta | Architecture design, LoRA injection, Newton experiments, hypernetwork implementation |
| Bhargavi | Dataset curation, VACE pipeline integration |
| Samhith Reddy | Training pipeline, SLURM scripts, evaluation |
| Sarayu | Related work, analysis, report writing |

---

## Method

### Architecture

```
Reference Images (N × 3 × H × W)
        ↓
CLIP ViT-H Encoder (frozen)  →  (N, 1024)
        ↓
Attention Pooling across N images  →  (1024,)
        ↓
Shared MLP Backbone  →  (hidden_dim,)
        ↓
Per-Block Weight Heads  →  LightLoRA {A_pred, B_pred} per layer
        ↓
Combine with frozen A_aux, B_aux  →  ΔW = A_aux × A_pred × B_pred × B_aux
        ↓
Inject into frozen Wan2.1-VACE  →  Personalized edited video
```

### LightLoRA Decomposition

Standard LoRA: `ΔW = A × B`

LightLoRA (ours): `ΔW = A_aux × A_pred × B_pred × B_aux`

- `A_aux`, `B_aux` — frozen orthogonal bases, shared across all subjects
- `A_pred`, `B_pred` — tiny subject-specific matrices predicted by the hypernetwork
- Rank = 4, ~614K parameters per model, <50KB per subject

### Base Model

**Wan2.1-VACE-1.3B** (`Wan-AI/Wan2.1-VACE-1.3B-diffusers`) — chosen over HunyuanVideo and CogVideoX for:
- Native video editing support (inpainting, masking, reference conditioning)
- 1.3B variant enables fast prototyping on a single V100
- Active LoRA ecosystem (VACE, Kiwi-Edit, SAMA)

---

## Setup

### Requirements

- Python 3.10+
- CUDA 12.6
- PyTorch 2.4+ with cu124

### Install

```bash
git clone https://github.com/vardhanreddy369/VideoEditPG.git
cd VideoEditPG
conda env create -f environment.yml
conda activate videoeditpg
```

Or with pip:

```bash
pip install -r requirements.txt
```

### Download Base Model

```bash
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B-diffusers \
  --local-dir ./models/wan_vace_1_3b
```

---

## Reproducing Results

### 1. Identity Generation

Generate human identity videos from text prompts using Wan2.1-VACE:

```bash
python tools/param_generator/inference_paramgen.py \
  --mode identity \
  --prompt "A young woman with curly hair dances in a studio" \
  --output_dir results/identity_generation
```

### 2. LoRA Injection Test

Verify LoRA can be injected into VACE without quality degradation:

```bash
python tools/param_generator/inference_paramgen.py \
  --mode lora_test \
  --lora_rank 4 \
  --output_dir results/lora_test
```

Expected: MSE ≈ 0.05, overhead < 5%, VRAM unchanged.

### 3. Train the Hypernetwork

```bash
python tools/param_generator/train_paramgen.py \
  --config configs/paramgen/train.yaml \
  --data_dir data/custom \
  --output_dir checkpoints/hypernet
```

On Newton HPC (SLURM):

```bash
sbatch tools/param_generator/newton_train_paramgen.sh
```

---

## Repository Structure

```
VideoEditPG/
├── tools/
│   └── param_generator/
│       ├── hypernet.py           # Hypernetwork architecture
│       ├── train_paramgen.py     # Training script
│       ├── inference_paramgen.py # Inference script
│       └── newton_train_paramgen.sh  # SLURM job script
├── configs/
│   └── paramgen/                 # Training configs
├── data/
│   └── custom/                   # Reference images for subjects
├── results/                      # Output videos
├── workspace/
│   ├── model_comparison_report.md    # Base model selection rationale
│   └── research_proposal_video_editing_pg.md  # Full research proposal
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Base model | Wan2.1-VACE-1.3B | Only open model with native video editing + LoRA support |
| LoRA decomposition | LightLoRA (rank 4) | <50KB per subject, minimal inference overhead |
| Encoder | CLIP ViT-H | Strong visual features, frozen — no extra training cost |
| Training loss | Diffusion reconstruction + L2 weight-space | Supervises in weight space, not just pixel space |

---

## Limitations

- VACE text-guided editing pipeline requires `ftfy` dependency — fix in progress
- Hypernetwork trained on faces only (CelebV-HQ); generalization to other subjects TBD
- Full end-to-end training not yet complete — current results show LoRA injection feasibility

---

## Related Work

- [HyperDreamBooth](https://arxiv.org/abs/2307.06949) (CVPR 2023) — hypernetwork for image personalization
- [Video2LoRA](https://arxiv.org/abs/2603.08210) (CVPR 2026) — parameter generation for video generation
- [VACE](https://github.com/ali-vilab/VACE) — Wan2.1 video editing backbone
- [HY-WU](https://arxiv.org/abs/2603.xxxxx) — parameter generation for image editing (Tencent)

---

## License

MIT
