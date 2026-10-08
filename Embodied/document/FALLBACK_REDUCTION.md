# Reducing NTP Fallback in LocateAnything's Hybrid Decoding — Experiment Log

Living log for the fallback-reduction work on branch `pbd-fallback-reduction`
(fork: `github.com/Haeryz/Eagle`). W&B project: **LocateAnything**.

## 1. Problem

LocateAnything decodes each box as one 6-token block in a single forward pass (MTP, Parallel Box Decoding).
In **Hybrid** mode a block is rejected and re-decoded token by token (NTP) when it is unreliable:

- **Spatial ambiguity**: a coordinate's top-1 probability is low *and* the top-k coordinate candidates are spread
  out (paper: top-1 < 0.7, spread > 80; **released code: top-1 < 0.9, spread > 60**,
  `eaglevl/utils/locany/generate_utils.py`).
- **Format irregularity**: the block is not a well-formed `<box> c c c c </box>` (or point / empty) frame.

The paper (arXiv 2605.27365) lists this as its main limitation: the model is SFT-only, and reducing *fallback
frequency* is left to future work. The paper never reports how often fallback happens. Its Table 12 shows how much
the MTP stream depends on it (Fast = MTP only, Hybrid = MTP + fallback, Slow = NTP only):

| Dataset (F1@mIoU) | Fast | Hybrid | Slow |
|---|---|---|---|
| COCO | 52.2 | 54.7 | 55.1 |
| LVIS | 47.0 | 50.7 | 52.6 |
| Dense200 | 46.8 | 61.3 | 61.5 |
| SROIE | 38.8 | 39.3 | 64.4 |
| RefCOCOg val / test | 70.8 / 72.5 | 73.4 / 74.8 | 72.4 / 73.8 |
| Throughput (BPS, H100) | 15.3 | 12.7 | 4.3 |

**Goal**: fewer fallbacks per box block, at equal or better Hybrid F1.
The fallback thresholds (0.9 / 60) stay **fixed** in every experiment. Loosening them would reduce fallbacks
trivially, without improving the model.

## 2. Methods (from the literature review in `source.md`)

### M1 — Grammar-constrained block decoding (inference only)
- **Reference**: Suresh et al., *DINGO: Constrained Inference for Diffusion LLMs*, NeurIPS 2025.
- **Targets**: format-irregularity fallbacks.
- **Mechanism**: a box block is a tiny regular language with three members: `<box> c c c c </box>`,
  `<box> c c </box> <null> <null>` (point) and `<box> none </box> <null>…` (empty). DINGO's dynamic program returns
  the most probable string in a regular language. For a fixed-length block whose positions are factorized given the
  context, that dynamic program reduces exactly to scoring each template by its summed best per-position
  log-probabilities, with coordinates arg-maxed over the 1001 coordinate tokens only. Malformed frames become
  impossible by construction. The ambiguity rule is unchanged (it is a shared function), so genuinely uncertain
  coordinates still fall back.
- **Code**: `constrained_box_decode()` in `eaglevl/utils/locany/generate_utils.py`, enabled with
  `generate_kwargs['constrained_block']` (eval flag `--constrained_block`). Default off.
- **Reuse note**: DINGO's official repo (`uiuc-focal-lab/DINGO`) contains only a README and demo GIFs, with no
  released code. The glue (about 40 lines) was therefore written from the paper.

### M2 — Sequence-level self-distillation + certainty forcing (LoRA training)
- **References**:
  - Kim & Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016.
  - Zhou, Gu & Neubig, *Understanding Knowledge Distillation in Non-Autoregressive Machine Translation*,
    ICLR 2020. Distilled targets are less multimodal, which is the main fix for NAT multimodality
    (Gu et al., ICLR 2018).
  - Chen et al., *dParallel: Learnable Parallel Decoding for dLLMs*, ICLR 2026. Certainty-forcing distillation.
- **Targets**: spatial-ambiguity fallbacks. When the annotations allow several valid "next boxes" (object order,
  spacing), each coordinate's marginal becomes bimodal, and the 6 tokens predicted independently in one pass
  inherit that ambiguity.
- **Mechanism**:
  1. The frozen base model decodes COCO train2017 images in **Slow (NTP) mode, greedily**. Its outputs become the
     targets, with one deterministic order and spacing, so the targets are near-unimodal.
     `evaluation/distill/generate_seqkd.py`.
  2. As in dParallel, wrong answers are filtered out: a sample is kept only if its boxes match COCO GT
     (per-category Hungarian matching, IoU ≥ 0.5, precision ≥ 0.8).
  3. Every image used by any eval set is excluded. LVIS-val and RefCOCOg draw images from COCO train, so this
     check matters.
  4. LoRA training uses the unchanged PBD objective L_ntp + L_mtp on these targets, **plus** dParallel's
     certainty-forcing term β·H(softmax(z/T)). It is applied only to MTP-block positions that the model already
     predicts correctly (argmax = target), with T = 0.5.
- **Code**: `certainty_forcing_loss()` and `mtp_label_indices()` in `eaglevl/model/locany/modeling_locateanything.py`,
  enabled with `--certainty_forcing_beta`. MTP positions are found where the position ids drop inside each packed
  sub-sample; this was unit-tested on a synthetic packed batch.
- **Ported from the authors' reference implementation** (`github.com/czg1225/dParallel`, `DLMTrainer.compute_loss`):
  the same temperature, the same correct-token mask, and the same mean over correct positions.
- **Hyperparameters taken from the reference, not tuned**: LoRA r = 32, lr 2e-5, cosine schedule, 5% warmup,
  weight decay 0.01, grad clip 1, 3 epochs.
  - β = 1, following the reference config for **Dream**, which is initialized from Qwen2.5, the same family as
    LocateAnything's decoder. The paper text says β = 2, as does the LLaDA config, so the reference code itself is
    inconsistent; this is noted rather than resolved by sweeping.
  - The repo's existing LoRA wrapper uses alpha = 2r, while dParallel uses alpha = r.
- **Ablation**: E2a trains on distilled targets with β = 0, isolating the distillation effect. E2b adds certainty
  forcing with β = 1.

### M3 — GRPO rewarding accuracy *and* fewer fallbacks (E4; the paper's stated future work)
- **References**: Perception-R1 (Yu et al., NeurIPS 2025): GRPO for detection with a Hungarian-matched reward.
  d1 / diffu-GRPO (Zhao et al., NeurIPS 2025): policy gradients for parallel decoders with factorized block
  likelihoods. Rex-Omni (Jiang et al., CVPR 2026): RL on a 3B coordinate-token detector removes SFT duplicate and
  misaligned-coordinate artifacts. The two-reward normalization follows LightningRL (arXiv 2603.13319, preprint, used
  for motivation only).
- **Targets**: what SFT cannot reach. Supervised targets say what the right box is, but not "emit a block that passes
  the reliability check". RL can reward that directly.
- **Ported from the Perception-R1 reference code** (`github.com/linkangheng/PR1`):
  - `pr1_detection_reward`: 0.25·format + 0.75·F1 + class + mean IoU over DETR-Hungarian-matched pairs (L1 + GIoU +
    class cost, IoU > 0.5; no FP/FN penalty, as in their detection script).
  - The GRPO loss −exp(logp − logp.detach())·A + β·k3-KL, with the reference being the same network with the adapter
    disabled.
  - Settings: β = 0.04, 8 generations per prompt, temperature 1.0, linear LR decay, grad clip 1.
- **Adaptations to PBD**:
  1. Rollouts use the model's own Hybrid decoding with coordinates *sampled* (not arg-maxed) in MTP blocks, so the
     group explores. The fallback rule is unchanged.
  2. A second reward r_par = 1 − (fallbacks / box blocks) is group-normalized separately and added to the accuracy
     advantage, so neither reward needs a hand-tuned weight.
  3. The rollout log-likelihood comes from **one teacher-forced pass in the PBD training layout**: the NTP stream
     scores fallback tokens, and each MTP step is re-created as an [anchor, mask×5] block at its anchor's positions.
     Only sampled tokens count as actions; deterministic structural tokens are excluded.
- **Verified**: re-scoring a rollout reproduced the sampling-time log-probs of all 198 actions (116 MTP coordinates,
  82 NTP tokens) with **max |Δ| = 0.0000** (fp32, `--check_consistency`). The RL gradient therefore uses exactly the
  decode-time contexts.
- **Deviation**: LR 1e-5 instead of PR1's 1e-6. PR1 fine-tunes all weights, while here only the LoRA adapter is
  trained, and LoRA's optimal LR is about 10× higher than full fine-tuning's (Biderman et al., *LoRA Learns Less and
  Forgets Less*, TMLR 2024). Steps are bounded by the single GPU: 150 steps × 2 prompts × 8 rollouts, initialized from
  E2b.
- **Code**: `eaglevl/train/grpo_pbd.py`. Rollout trace and coordinate sampling are in `generate()` / `decode_bbox_avg`.

### Instrumentation (needed to measure fallback at all)
`generate()` now records, per sample: box blocks, fallbacks split by reason (format / ambiguity), and the
coordinate top-1 probability and entropy. These go into each prediction's `decode_stats`.
`evaluation/tools/summarize_fallback.py` aggregates them and logs to W&B.
Decoding choices are unchanged: the baseline uses the release's decoding logic byte-for-byte (diffed against the
HF checkpoint's `generate_utils.py`).

## 3. Evaluation protocol

- **Hardware**: a single RTX 2080 Ti (11 GB, Turing). There is no bf16 GEMM, flash-attn or MagiAttention, so
  everything runs in fp16 with `sdpa`.
- **Subsets**: fixed, seeded subsets (`evaluation/tools/make_subsets.py`, seed 0): COCO 500, LVIS 500,
  RefCOCOg val 500, RefCOCOg test 500, Dense200 (all 200), SROIE (all 360).
- **Decoding**: the release's eval settings (temperature 0.7, top-p 0.9, repetition penalty 1.1), with a
  **per-sample seed**, so compared runs see the same random stream for the same input.
- **Metric**: F1 averaged over IoU 0.50:0.95 (`evaluation/metrics/other_metric.py`), applied uniformly to all six
  sets.
  - COCO/LVIS are additionally re-scored with the paper's official FastEvaluate pipeline (see section 5), which is
    the version comparable to the paper.
  - RefCOCOg, Dense200 and SROIE use the same metric as the paper.
- **Fallback rate**: switches to NTP divided by MTP box blocks, split by reason.
- **Comparisons**:
  - Δ against our reproduced baseline (E0) is the valid, same-hardware, same-subset comparison.
  - Δ against the paper is approximate (subset, fp16, 11 GB GPU).
  - BPS is relative only, since the paper's numbers are on an H100.

## 4. Experiments

| ID | Change | Status |
|---|---|---|
| E0 | Released model, Hybrid (and Fast) | Hybrid done; Fast queued |
| E1 | E0 + M1 constrained blocks | queued |
| E2a | Self-distilled LoRA, β = 0 | done (Hybrid, 6/6) |
| E2b | Self-distilled LoRA + certainty forcing, β = 1 | done (Hybrid, 6/6) |
| E3 | E2b + M1 | done (Hybrid, 6/6) |
| E4 | E2b + GRPO (accuracy + parallelism reward) | queued |

Results go in section 5. Every experiment is reported as hypothesis → method → code change → result against E0.

## 5. Results

### E0: reproduced baseline (released LocateAnything-3B, Hybrid mode)

Fallback columns are per-sample (macro) means: % of a sample's MTP box blocks that fell back to NTP. Runaway counts
samples that hit the 8192-token cap in a repetition loop.

| Subset | F1@mIoU (ours) | Paper Hybrid | Fallback | Ambiguity | Format | Coord entropy | Coord top-1 | Runaway |
|---|---|---|---|---|---|---|---|---|
| RefCOCOg val (500) | 74.12 | 73.4 | 19.0% | 19.0% | 0.00% | 2.17 | 0.27 | 0 |
| RefCOCOg test (500) | 78.96 | 74.8 | 13.8% | 13.8% | 0.00% | 2.10 | 0.28 | 0 |
| COCO (500) | 63.41* | 54.7* | 18.5% | 17.7% | 0.80% | 1.66 | 0.40 | 1 |
| LVIS (500) | 51.06* | 50.7* | 22.7% | 21.9% | 0.75% | 1.83 | 0.38 | 2 |
| Dense200 (200) | 59.52 | 61.3 | 26.1% | 26.1% | 0.04% | 1.32 | 0.52 | 0 |
| SROIE (360) | 39.17 | 39.3 | 6.4% | 6.4% | 0.00% | 1.00 | 0.59 | 8 |

\* Different metric pipeline from the paper's COCO/LVIS numbers (see section 3).

- **The reproduction is faithful** where the metric matches: RefCOCOg val +0.7, Dense200 −1.8, SROIE −0.1 against
  the paper. These are subset-sized samples on fp16 / single GPU.
- **Noise floor**: re-running E0 with numerically different but mathematically identical ViT kernels moved
  F1 ≤ 0.12 and per-sample fallback ≤ 0.4 points (RefCOCOg val, COCO, LVIS). A method effect must clearly exceed
  this.
- **Fallback is overwhelmingly ambiguity-driven** (≥ 96% of fallbacks on every subset). Dense layouts trigger it
  most (Dense200: one block in four).

### E2a: sequence-level self-distillation (LoRA, β = 0) vs E0, Hybrid mode

**Hypothesis**: training the MTP stream on the model's own deterministic (Slow-mode, greedy, GT-filtered) outputs
makes per-coordinate targets unimodal (Zhou et al., ICLR 2020), so fewer coordinates trip the ambiguity rule.

| Subset | F1 E0 → E2a | Δ F1 | Fallback (per image) E0 → E2a | Δ | Pooled fallback E0 → E2a | Coord entropy E0 → E2a |
|---|---|---|---|---|---|---|
| RefCOCOg val | 74.12 → 74.66 | +0.54 | 19.0 → 22.2 | +3.2 | 19.0 → 22.2 | 2.17 → 1.37 |
| RefCOCOg test | 78.96 → 80.12 | +1.16 | 13.8 → 17.0 | +3.2 | 13.8 → 17.0 | 2.10 → 1.33 |
| COCO | 63.41 → 64.21 | +0.80 | 18.5 → 16.3 | −2.2 | 19.4 → 21.7 | 1.66 → 0.66 |
| LVIS | 51.06 → 52.64 | +1.58 | 22.7 → 22.8 | +0.1 | 14.2 → 18.3 | 1.83 → 0.78 |
| **Dense200** | **59.52 → 64.62** | **+5.10** | **26.1 → 21.3** | **−4.8** | 26.1 → 23.1 | 1.32 → 0.73 |
| SROIE | 39.17 → 41.31 | +2.14 | 6.4 → 5.5 | −0.9 | 8.5 → 7.0 | 1.00 → 0.62 |

- **Accuracy improves on every subset.** Dense200 Hybrid (64.6) now exceeds the paper's own Slow / NTP-only mode
  (61.5).
- **Fallback falls where it matters most**: dense scenes, −4.8 points per image, and the COCO-domain typical image,
  −2.2. It **rises on RefCOCOg** (+3.2), a prompt type ("locate a single instance") absent from the distillation
  data.
- **Mechanism check**: coordinate entropy drops 37–60% everywhere, so the targeted signal (multimodal coordinate
  marginals) did move.

### E2b: self-distillation + dParallel certainty forcing (LoRA, β = 1) vs E0 and E2a, Hybrid mode

**Hypothesis**: on top of E2a, minimizing the entropy of already-correct MTP predictions (dParallel, ICLR 2026)
pushes coordinate confidence up, so fewer coordinates sit in the low-confidence region the fallback rule inspects.
**Training-side mechanism check**: on training batches, mean MTP-coordinate top-1 rose from 0.33 to 0.69, and the
share of coordinates below 0.9 fell from 97% to 72% (W&B run `train-E2b`).

| Subset | F1 E0 / E2a / **E2b** | Δ F1 (E2b − E0) | Fallback per image E0 / E2a / **E2b** | Δ (E2b − E0) | Pooled E0 / E2b | Format fallback E0 / E2b |
|---|---|---|---|---|---|---|
| RefCOCOg val | 74.12 / 74.66 / **74.40** | +0.28 | 19.0 / 22.2 / **22.4** | +3.4 | 19.0 / 22.4 | 0.00 / 0.00 |
| RefCOCOg test | 78.96 / 80.12 / **80.04** | +1.08 | 13.8 / 17.0 / **18.0** | +4.2 | 13.8 / 18.0 | 0.00 / 0.00 |
| COCO | 63.41 / 64.21 / **64.22** | +0.82 | 18.5 / 16.3 / **15.9** | −2.6 | 19.4 / 21.7 | 0.80 / 0.69 |
| LVIS | 51.06 / 52.64 / **52.56** | +1.50 | 22.7 / 22.8 / **21.7** | −1.0 | 14.2 / 25.2 | 0.75 / 1.34 |
| **Dense200** | 59.52 / 64.62 / **64.37** | **+4.85** | 26.1 / 21.3 / **20.1** | **−6.0** | 26.1 / 20.6 | 0.04 / 0.08 |
| SROIE | 39.17 / 41.31 / **41.28** | +2.11 | 6.4 / 5.5 / **5.8** | −0.6 | 8.5 / 8.1 | 0.00 / 0.00 |

- **F1 improves on all six subsets.** Certainty forcing keeps distillation's gains (within ±0.3 of E2a).
- **Fallback falls on 4 of 6 subsets** and is largest where the paper's limitation bites: Dense200 −6.0 points per
  image (a 23% relative cut) and −5.5 pooled. Certainty forcing adds −1.3 (Dense200), −0.4 (COCO) and −1.1 (LVIS)
  over distillation alone.
- **RefCOCOg fallback rises** (+3.4 / +4.2). Out of domain (no referring prompts in the distillation data),
  certainty forcing barely sharpens (top-1 0.45 → 0.48, far below the rule's 0.9 gate), so the misfiring described
  below is not bypassed.
- **Format fallbacks grow slightly on LVIS** (0.75% → 1.34%). Sharper predictions commit harder to malformed frames
  on long category lists; this is what M1 (E3) targets.

### E3: E2b + M1 grammar-constrained block decoding, Hybrid mode

**Hypothesis**: the format fallbacks that remain, and that grow with sharpening (LVIS 0.75% → 1.34%), are removed by
decoding each box block as the most probable *legal* frame (DINGO, NeurIPS 2025). Ambiguity fallbacks are
unaffected.

| Subset | F1 E0 / E2b / **E3** (universal) | Δ F1 E3 − E0 | Fallback E0 / E2b / **E3** | Δ E3 − E0 | Format fallback E2b → E3 |
|---|---|---|---|---|---|
| RefCOCOg val | 74.12 / 74.40 / **74.42** | +0.30 | 19.0 / 22.4 / **22.4** | +3.4 | 0.00 → 0.00 |
| RefCOCOg test | 78.96 / 80.04 / **80.06** | +1.10 | 13.8 / 18.0 / **18.0** | +4.2 | 0.00 → 0.00 |
| COCO | 63.41 / 64.22 / **64.67** | +1.26 | 18.5 / 15.9 / **15.8** | −2.7 | 0.69 → 0.62 |
| LVIS | 51.06 / 52.56 / **53.07** | +2.01 | 22.7 / 21.7 / **21.2** | −1.4 | **1.34 → 0.76** |
| **Dense200** | 59.52 / 64.37 / **65.33** | **+5.81** | 26.1 / 20.1 / **19.9** | **−6.2** | 0.08 → 0.06 |
| SROIE | 39.17 / 41.28 / **41.28** | +2.10 | 6.4 / 5.8 / **5.7** | −0.7 | 0.00 → 0.00 |

- **Mechanism check**: format fallbacks fall where they existed (LVIS −44%, COCO −10%), and ambiguity fallbacks are
  unchanged. This is as designed.
- **The remaining format triggers** come from blocks with P(`<box>`) < 0.6, which M1 deliberately leaves to the
  original decoder (e.g. a block emitting the text token `None` instead of the `none` empty-box token).
- Under the official FastEvaluate metric, E3 equals E2b on COCO/LVIS (below), so M1's F1 effect is metric-dependent.
  Its fallback effect is not.

### COCO / LVIS with the paper's official pipeline (FastEvaluate)

The universal metric above is not the one the paper uses for COCO/LVIS. The saved predictions were therefore
re-scored with the repo's own `convert_coco_lvis_to_standard_format.py --positive_only` + `fastevaluate`
(C++ extension in `evaluation/fastevaluate`), unchanged, with the GT JSON restricted to the 500 subset images
(`evaluation/tools/fasteval_subset.py`). No re-inference was needed.

| | COCO F1@mean | Δ vs E0 | COCO F1@0.5 | COCO F1@0.95 | LVIS F1@mean | Δ vs E0 | LVIS F1@0.5 | LVIS F1@0.95 |
|---|---|---|---|---|---|---|---|---|
| Paper Hybrid (full val) | 54.7 | | 70.1 | 19.3 | 50.7 | | 62.3 | 31.1 |
| **E0 reproduced** | **55.31** | — | 72.41 | 23.49 | **53.18** | — | 70.34 | 39.54 |
| E2a | 56.85 | **+1.54** | 74.02 | 24.35 | 53.88 | **+0.70** | 71.65 | 40.67 |
| E2b | 56.55 | +1.24 | 73.42 | **24.90** | 53.31 | +0.13 | 71.31 | **42.69** |
| E3 | 56.55 | +1.24 | 73.63 | 24.80 | 53.31 | +0.13 | 71.16 | 42.64 |

- **The reproduction matches the paper** on COCO (55.3 vs 54.7). The LVIS subset of 500 images scores higher than
  the full 19.6k-image long-tail val (53.2 vs 50.7), so treat LVIS absolute numbers as approximate.
- **Gains hold under the official metric** but are smaller on LVIS than the universal metric suggested.
- **The largest effect is at strict IoU**: certainty forcing raises F1@0.95 by +3.2 (LVIS) and +1.4 (COCO), which is
  consistent with sharper coordinate distributions giving more precise edges.
- **M1 adds nothing over E2b** on COCO/LVIS under this metric; its +0.5 under the universal metric does not carry
  over.

**Diagnosis of the new RefCOCOg fallbacks** (`evaluation/tools/diagnose_ambiguity.py`, the 41 RefCOCOg-val rows
where E2a newly falls back; fp32 CPU replay logging the top-k coordinates at every trigger):

| | Rows with a trigger | Triggers | Genuinely bimodal | Dominant cluster + distant low-mass tail | Mean top-k coord mass |
|---|---|---|---|---|---|
| E0 | 7 | 8 | 2 | 6 | 0.52 |
| E2a | 29 | 35 | 7 | 28 | 0.73 |

Example E2a trigger: `[990: 0.40, 988: 0.26, 998: 0.20, 852: 0.04]`, i.e. 86% of the mass in one tight cluster,
yet it fires. The release rule (top-1 < 0.9 ∧ >1 coordinate in the top-k ∧ max−min of top-k values > 60) is
**probability-blind in its spread test**. After distillation sharpens a coordinate into a narrow peak, fewer
neighbouring bins fill the top-4, so an improbable distant bin enters the top-k and trips the spread condition.
The increase in fallbacks is therefore mostly the reliability check misfiring on *confident* predictions, not new
ambiguity. Genuine two-object bimodality is 7 of 35 triggers.

**Implication for the next step**: either push top-1 above the rule's 0.9 gate (certainty forcing, E2b), or replace
the mass-blind heuristic with a principled, mass-aware criterion such as EB-Sampler's entropy bound
(Ben-Hamu et al., NeurIPS 2025). The latter must be applied to the baseline as well and validated as an error
detector (it should fire on blocks whose Fast-mode box is wrong), not tuned to lower the fallback count.

## 7. Next steps (next week)

1. **Evaluation scale.** COCO, LVIS and RefCOCOg val/test were evaluated on fixed, seeded 500-sample subsets
   (Dense200 and SROIE in full), because of single-GPU time (~1 h 50 min per configuration on the subsets vs.
   ~9–10 h on the full sets).
   - Run the baseline and the final best model on the **full** COCO and RefCOCOg val/test sets (~2.5 h per model),
     and on full LVIS val if a larger GPU is available (~6.5 h per model on the 2080 Ti).
   - Report **paired bootstrap 95% confidence intervals** for every Δ (resampling images; same images across
     models), so each improvement is stated with its significance.
2. **Fix the reliability check itself.** The diagnosis (section 5) shows the release rule's top-k spread test
   ignores probability mass and misfires on sharpened, confident predictions.
   - Evaluate a mass-aware criterion, EB-Sampler's entropy bound (Ben-Hamu et al., NeurIPS 2025), against the
     release rule **as a wrong-box detector at an equal fallback budget**, using the per-block features logged in
     the Fast-mode runs (`decode_stats['blocks']`).
   - Apply it to the baseline as well, so any gain is not just threshold loosening.
3. **Close the RefCOCOg gap.** RefCOCOg fallback rose because the distillation data contained only multi-category
   detection prompts. Add self-distilled *referring* prompts (RefCOCO/+/g train splits, disjoint from the eval
   images) to the distillation set, then re-check whether the out-of-domain sharpening issue disappears.
4. **RL at scale (E4 follow-up).** E4 is bounded by one GPU (150 steps × 2 prompts × 8 rollouts). Scale the number
   of prompts and steps, and add dense-scene prompts, where the fallback reward has the most signal.
5. **Speed.** Report BPS on the paper's hardware (H100), since 2080 Ti throughput is only relative. Measure the
   end-to-end speed-up from fewer fallbacks.

## 8. Problems encountered and how they were resolved

| # | Problem | Resolution |
|---|---|---|
| 1 | `LocateAnything.pdf` in the repo is a Git-LFS pointer (133 bytes), not the paper | Read the paper from arXiv HTML (2605.27365) |
| 2 | Disk 100% full (0 B free) | `uv cache clean` (freed 15.3 GB). Adapter-only LoRA saves (`--save_lora_adapter_only`) and streaming tar extraction keep usage low |
| 3 | The local clone was shallow, so the push to the fork was rejected (missing parent object) | `git fetch --unshallow origin`, then a plain `git push fork main` (no `gh`) |
| 4 | Single Turing GPU: no bf16, flash-attn or Magi; the training ViT was hard-forced to flash-attn; SDPA masks were hard-coded bf16 | The ViT falls back to its existing `sdpa` path; masks follow the activation dtype; fp16 AMP keeps trainable params in fp32 |
| 5 | Is fp16 safe for a bf16-trained Qwen2.5? | Gate test on 20 RefCOCOg samples, fp16 vs emulated bf16: same fallback rate (35%), entropy (2.30 vs 2.28) and top-1 (0.248 vs 0.252); 14/20 identical outputs, the rest sampling flips. fp16 is 2.3× faster, so fp16 is used |
| 6 | The HF xet download stalled repeatedly | Resumable `curl` with a speed floor; SHA-256 verified against the Hub |
| 7 | Dense200 OOM in the ViT: the `sdpa` path built a dense [S,S] mask over up to 25,600 patches and passed 3-D tensors, so PyTorch fell back to the math kernel (40 GB allocation) | Per-image SDPA with 4-D inputs, which enables the memory-efficient kernel. Exact: max abs diff 0.0 vs the masked version |
| 8 | Dense200 OOM at prefill: the LM head projected all ~6.4k prompt tokens over the 152k vocab in fp32 (3.7 GB) | `logits_to_keep`: only the last block's logits are computed, which is all that generation reads |
| 9 | Fixes 7–8 changed kernels mid-baseline (same math, different fp16 kernels) | All evals restarted on identical code. The earlier runs are kept in `work_dirs/results_oldvit/` as a kernel-noise reference |
| 10 | dParallel's β differs between the paper (2) and the reference configs (LLaDA 2, Dream 1) | Used the Dream setting (Qwen2.5-initialized, closest to our decoder) and documented it; no sweep |
| 11 | The release's thresholds (0.9/60) differ from the paper (0.7/80) | Kept the released code's values, since they are what the checkpoint ships with |
| 13 | My E4 trace edit (commit 15a4cc1) dropped the `num_box_blocks` guard, so box blocks were not counted in normal evals (decoding unaffected) | Fixed in a251ab2. For the two affected E2a files, box blocks == `num_boxes` (every `<box>` is opened by an MTP step; verified on all 2560 E0 samples) recovers the denominator exactly |
| 14 | Pooled fallback ratios are dominated by a few repetition-loop samples that hit the token cap | Per-image (macro) averages are the primary metric; pooled values and runaway counts are reported alongside |
| 12 | Coordinate top-1 is naturally low (~0.25–0.4) because probability mass spreads over neighbouring ordinal bins, so "top-1 < 0.9" is ~95% everywhere and uninformative | Mechanism checks use coordinate entropy and the actual ambiguity-trigger rate instead |

### Early observations (old-kernel runs, `results_oldvit/`, before restart)

| Subset | Hybrid F1@mIoU | Paper Hybrid | Fallback / block | Format | Ambiguity |
|---|---|---|---|---|---|
| RefCOCOg val | 74.0 | 73.4 | 19.2% | 0.0% | 19.2% |
| RefCOCOg test | 78.9 | 74.8 | 13.0% | 0.0% | 13.0% |
| COCO | 63.5* | 54.7* | 23.9% | 0.6% | 23.3% |
| LVIS | 51.1* | 50.7* | 13.1% | 0.3% | 12.8% |

\* Different metric pipeline from the paper (see section 3).

**Takeaway so far**: almost all fallbacks are **ambiguity-triggered**, and format fallbacks are under 1%. So M1 can
remove at most about 0.6 points of fallback rate, and the training-side method M2, which targets coordinate
ambiguity, is the one that can move the metric substantially. This matches the CLAUDE.md requirement to verify the
mechanism: we measured which trigger dominates before expecting either method to help.
