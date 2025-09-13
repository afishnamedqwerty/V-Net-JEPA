# H-Net V-JEPA: Hierarchical, Dynamic Chunking for End-to-End Video Representation Learning

H-Net V-JEPA replaces fixed tubulet tokenization in V-JEPA2 with a hierarchical, dynamic, end-to-end trainable pipeline. Raw video is encoded by a lightweight 3D CNN, routed into variable-sized semantic chunks with native sparse attention, downsampled via learned pooling, enriched with adaptive positional encodings, and then processed by a ViT backbone. The stack is trained jointly with energy-based objectives and VICRegL regularization, and supports multimodal alignment and action-conditioned post-training for robotic planning.

## Repository Structure

- models/hnet/
  - encoder.py: Low-level 3D CNN (ResNet-18 3D variant) preserving spatiotemporal structure; outputs h ∈ R^{T'×H'×W'×D}.
  - routing.py: SparseRouting with NativeSparseAttention for dynamic chunking, soft assignments, and prototype-based aggregation.
  - downsampler.py: LearnedAttentionPooling (query-token attention) producing pooled chunks z′ with centroids and sizes.
  - pos_enc.py: Adaptive positional encoding MLP using chunk centroids and sizes.
  - dechunker.py: Optional smoothing/upsampling (EMA + STE confidence) for feature reconstruction.
- models/vjepa/
  - vit.py: HNetViT composes encoder → routing → downsampler → adaptive pos → ViT. Integrates activation recomputation and zig-zag ring attention. EMA target maintained and broadcast across ranks.
  - predictor.py: Lightweight transformer predictor for masked chunk forecasting.
- losses/
  - energy.py: Energy loss for E(x, y) and modality-aligned variants.
  - vicregl.py: VICRegLLoss for chunk-level invariance, variance, covariance regularization.
  - auxiliaries.py: Ratio/entropy/boundary regularizer hooks/interfaces.
- trainers/
  - pretrain.py: SSL masked prediction with EMA targets and VICRegL.
  - language_align.py: NT-Xent between pooled video/text + Energy regularizer.
  - probe_train.py: Attentive probe with BCE logits + Energy coherence regularizer.
  - action_posttrain.py: Action-conditioned training using cross-attention fusion and conditioned energy; CEM planning utility.
- utils/
  - misc.py: Padding/masking utilities, STE ops, NT-Xent, CEMPlanner, CrossAttentionFuser, get_device, set_seed.
  - distrib.py: DDP helpers, rank/world utils, activation recomputation wrapper (checkpoint_sequential), and zig-zag ring attention.
- configs/
  - pretrain_ssv2.yaml: Baseline hyperparameters for SSv2.
  - deepspeed_zero2.json (recommended): Reference ZeRO-2 configuration.
- main.py: Orchestrates sequential training phases with per-phase checkpointing and pipeline manifest.

## Core Architecture

Input video x ∈ R^{B×T×H×W×3} (e.g., T=32, H=W=224).

1) Low-Level Encoder (models/hnet/encoder.py)
- ResNet-18 3D variant preserves temporal resolution while spatially downsampling.
- Output: h ∈ R^{B×T′×H′×W′×D} with D=256.

2) Routing with Native Sparse Attention (models/hnet/routing.py)
- Flatten h → h_flat ∈ R^{B×N_f×D}, N_f=T′×H′×W′.
- NativeSparseAttention compresses/localizes; SparseRouting cross-attends to K prototypes to produce soft assignments p and chunk embeddings z.
- Returns: z ∈ R^{B×M×D} (variable M) and p_full ∈ R^{B×N_f×K}.

3) Semantic Downsampling (models/hnet/downsampler.py)
- LearnedAttentionPooling reduces M→M′ (e.g., 196) using query tokens and attention.
- Computes centroids/sizes from assignments and grid positions; can emit pooled_mask.
- Returns: z′ ∈ R^{B×M′×D}, centroids ∈ R^{B×M′×3}, sizes ∈ R^{B×M′}, pooled_mask (optional).

4) Adaptive Positional Encoding (models/hnet/pos_enc.py)
- pos = MLP([centroid(3D), size(1D)]) ∈ R^D; z′′ = z′ + pos.

5) ViT Backbone (models/vjepa/vit.py)
- Processes z′′ via Transformer layers with batch_first attention and key_padding_mask (True = pad).
- Activation recomputation: Transformer blocks wrapped with torch.utils.checkpoint via utils.distrib.checkpoint_sequential to reduce activation memory (no engine-level double checkpointing).
- Zig-zag ring attention: Under multi-rank runs, attention uses utils.distrib.zigzag_ring_attention:
  - Circulates K,V around ranks in an alternating pattern (zig/zag), computes attention at each step, accumulates, and averages outputs for correctness.
  - Falls back to local attention when single rank.

6) Predictor (models/vjepa/predictor.py)
- Predicts masked chunk latents for SSL and future forecasting.

EMA Target
- Momentum EMA updates occur on all ranks; EMA parameters are broadcast from rank 0 each update to maintain identical targets across GPUs.

## Training Phases

1) Self-Supervised Pretraining (trainers/pretrain.py)
- Forward x to get contextualized chunks z_ctx; update EMA target and compute z_tgt from EMA model.
- Mask chunk indices deterministically per epoch/rank; predictor forecasts masked z_tgt from context.
- Loss: Energy(z_pred_masked, z_tgt_masked) + λ VICRegL(z_ctx, z_tgt).
- Supports gradient accumulation and DDP/DeepSpeed; uses DistributedSampler and per-epoch seeding.

2) Language Alignment (trainers/language_align.py)
- Pooled video embeddings aligned with text via NT-Xent + Energy regularizer (swap DummyTextEncoder for CLIP/BERT).
- Encourages coherent cross-modal semantics.

3) Attentive Probe Training (trainers/probe_train.py)
- Linear probe over pooled chunks with BCE logits; Energy regularization on salient chunks; optionally unfreeze routing.

4) Action-Conditioned Post-Training (trainers/action_posttrain.py)
- Cross-attention fuses action tokens and chunk embeddings; predict future chunk latents and minimize conditioned Energy.
- CEMPlanner provided for action trajectory optimization at inference.

Main Pipeline
- main.py sequences phases and writes a pipeline manifest with per-phase checkpoints:
```
python main.py --work_dir outputs --sequence pretrain,language,probe,action
```
bbygirl a friend of mine is discussing placing satellites in a polar orbit around the moon for spectral analysis of ideal mining locations
## Distributed Training

Key principles (see distributed_training.md for deep dive):
- Data Parallelism (DDP) and optional ZeRO-2 (DeepSpeed config).
- Activation Recomputation:
  - Transformer layers are wrapped via torch.utils.checkpoint (utils.distrib.checkpoint_sequential) to reduce activation memory with modest compute overhead.
  - Avoid double-checkpointing; keep recomputation at the layer level in PyTorch.
- Zig-Zag Ring Attention (utils.distrib.zigzag_ring_attention):
  - Rank-local ring circulates K,V; alternates directions each step and averages accumulated outputs for correctness.
  - Reduces dependence on global all-to-all; complements local sparsity in H-Net routing and supports sequence/context sharding.
  - Fallback to standard attention when single GPU.
- Determinism:
  - Use DistributedSampler with set_epoch. Mask sampling uses a per-epoch seeded RNG to produce identical masks per-rank.
  - EMA synchronization: target encoder parameters are broadcast after updates to avoid inter-rank drift.

Launch Examples
- DDP torchrun:
```
torchrun --nproc_per_node=8 --master_port=29500 \
  main.py --work_dir outputs --sequence pretrain,language,probe,action
```
- DeepSpeed ZeRO-2 (add configs/deepspeed_zero2.json):
```
deepspeed --num_gpus=8 main.py --work_dir outputs \
  --sequence pretrain,language,probe,action \
  --use_deepspeed 1 --deepspeed_config configs/deepspeed_zero2.json
```

## Datasets and Batches

- SSv2: pretraining, alignment, and probing.
- Droid: action-conditioned post-training and planning.
**does it make more sense to implement saliency-weighted pooling (class activation mapping) given saliency labels in the batch?**

Batch schema:
- batch['video']: B×T×H×W×3 (normalized).
- Optional:
  - batch['text']: list/ids for language alignment.
  - batch['saliency']: B×M′ (per pooled chunk labels).
  - batch['actions']: B×A×7 (robotic action sequences).

## Installation

Python 3.10+, CUDA-enabled PyTorch. Required deps (partial):
- torch, torchvision, torchaudio (CUDA)
- torch-geometric (k-NN graphs; install with matching CUDA wheels)
- deepspeed (optional ZeRO)
- numpy, tqdm, pyyaml, pytest (dev)

Install:
```
pip install -r requirements.txt
# For torch-geometric, follow official installation instructions for your CUDA version.
```

## Testing

Run basic tests:
```
pytest -q
```

## Design Notes

- Soft routing with p improves robustness to noisy boundaries.
- Learned pooling retains semantics while controlling sequence length (M′).
- Adaptive positional encoding integrates geometry/scale information.
- Zig-zag ring attention complements local sparsity and enables sequence/context scaling under multi-GPU settings.
- Activation recomputation is key to fit long sequences/large models in memory on A100/H100.

## Roadmap

- Optional production TP/SP via TransformerEngine/Megatron-Core.
- FlashAttention2 integration where applicable.
- Enhanced pooling (learned TopK/graph pooling).
- Stronger language encoders and expanded planning.

## License

Apache 2.0.
