# AdamW vs Muon: Diagnosing Optimization During Fine-Tuning

A lightweight study of how AdamW and Muon differ when fine-tuning a small
pretrained LM on SST-2. The training and evaluation loop is written by hand;
only the model, tokenizer and dataset come from libraries.

**Premise.** The two optimizers differ in *geometry*, not just tuning. AdamW
normalizes coordinate-wise, so its update inherits the gradient's spectral
concentration. Muon orthogonalizes the momentum matrix — steepest descent under
the spectral norm, i.e. the linear minimization oracle `argmax_{‖Δ‖₂≤1} ⟨M,Δ⟩ =
UVᵀ` — so every singular value of the step is ≈1. That is a falsifiable claim,
so the primary probes here are geometric rather than accuracy-based; SST-2
accuracy saturates and cannot separate these optimizers.

Full write-up: [`REPORT.md`](REPORT.md).

## Headline results

| | AdamW (lr 1e-4) | Muon (lr 3e-4) |
|---|---|---|
| dev / test / OOD accuracy | .920 / .918 / .863 | .922 / .914 / .866 |
| stable rank of ΔW (per layer) | 1.0 – 7.4 | 47 – 205 |
| spectral entropy of ΔW | .61 – .93 | .91 – .99 |
| dev acc at 30× optimal LR (3e-3) | 0.546 (chance = 0.509) | 0.845 |
| s/step (training only) | 0.567 | 1.237 |

Three findings, in order of how well they are supported:

1. **Geometry separates cleanly.** Muon's cumulative weight change is 20–40×
   higher stable rank than AdamW's, consistently across depth. At the deepest
   tracked matrix AdamW's ΔW has stable rank 1.0 — its entire cumulative update
   to that layer lies in essentially one direction.
2. **Muon is markedly more LR-robust**, but only above the optimum; it is worse
   than AdamW at low LR (.785 vs .864 at 1e-5). AdamW is 2.18× faster per step.
3. **Accuracy is a null.** All three splits sit inside one seed standard
   deviation. Sharpness is also a null, and the three estimators disagree on the
   sign — see the report.

## Running it

```bash
# Colab (T4), one cell:
%run adamw_vs_muon_sst2.py
# or locally, CUDA required:
pip install torch transformers datasets matplotlib
python adamw_vs_muon_sst2.py
```

Everything is in one file. ~3.5 h end to end on a free T4:

| stage | runs | time |
|---|---|---|
| LR sweep | 12 × 200 steps | ~36 min |
| main runs (3 seeds, full curvature probes) | 6 × 400 steps | ~75 min |
| momentum ablation (4 seeds) | 16 × 400 steps | ~100 min |

Completed runs are cached as `results/<tag>.json` and skipped on re-execution, so
an interrupted session costs one run rather than the session. A cached run whose
curvature probe failed is recomputed rather than silently leaving a hole.

Outputs land in `results/`: one JSON per run, four figures under
`results/figures/`, and `results/summary.md` with all three tables.

A self-test runs first (~5 s) and validates every estimator against a closed
form before any of them are trusted on a real model: Newton–Schulz collapsing a
badly conditioned spectrum, `λ_max` against the exact top eigenvalue of a known
quadratic, the Hutchinson trace against the exact trace, and a double-backward
through attention. If it fails, nothing downstream is meaningful.

## Experimental design

**Data.** SST-2 test labels are hidden, so 2 000 examples are held out from
`train` as the dev set used for *all* model selection; the official `validation`
split (872) is touched exactly once, as test. OOD evaluation uses
`rotten_tomatoes` (same task, longer reviews).

**Fairness contract.** Both arms use the *identical* parameter split — hidden
matrices to the optimizer under test, embeddings / classifier head / norms /
biases to AdamW at a fixed LR. The only quantity that varies is which optimizer
updates the matrices. Each arm gets its own LR sweep and we compare
**best-vs-best**: comparing at one shared LR measures tuning, not the optimizer.
The sweep in this repo shows why — at 3e-3 you would "prove" Muon is vastly
better, at 1e-5 the reverse.

**Muon scaling.** Uses the Moonlight convention `0.2·√max(m,n)`, which matches
the update's per-entry RMS to AdamW's (an orthogonalized `O` has
`‖O‖_F ≈ √min(m,n)`, so entry RMS is `1/√max(m,n)`). This is what makes a shared
LR grid meaningful. The original `√max(1, m/n)` convention would shift Muon's
optimum roughly 8× lower.

**Caveat not hidden.** The classification head is randomly initialized, so early
steps are head-fitting rather than fine-tuning.

## What is measured, and why

**Geometry** — stable rank `‖ΔW‖²_F/‖ΔW‖²₂` and spectral entropy of the
cumulative `ΔW = W_t − W₀`, plus per-layer relative update norm. These test the
orthogonalization hypothesis directly.

**Sharpness**, three ways, because they probe different things and can disagree:
`λ_max(H)` by power iteration on Hessian-vector products (one stiff direction);
`tr(H)` by Hutchinson (average curvature); and adaptive worst-case sharpness
`max_{‖ε/|θ|‖≤ρ} L(θ+ε) − L(θ)` (a reachable ball, and reparameterization
invariant). Plus a filter-normalized 1D loss profile.

**Gradient noise** — `cos(g_t, g_{t−1})`, the mechanistic bridge to the momentum
ablation: momentum buys variance reduction, and buys less when consecutive
gradients are already aligned.

**Cost** — wall-clock per step, so Muon's Newton–Schulz overhead is priced in
rather than hidden by comparing step counts.

## Implementation notes

Three things that are easy to get silently wrong, all handled here:

- **Newton–Schulz requires the Frobenius pre-normalization.** The quintic only
  contracts for singular values ≤ 1; above that it diverges. `‖G‖₂ ≤ ‖G‖_F`
  gives the guarantee for free.
- **Under fp16 you must unscale before clipping.** Clipping a 65536×-scaled
  gradient against a threshold of 1.0 silently clips everything to 1/65536 of
  what you intended — no error, just worse results.
- **Curvature probes need the SDPA math backend.** Flash and mem-efficient
  attention have no double-backward implementation, so `create_graph=True` fails
  with `derivative for aten::_scaled_dot_product_efficient_attention_backward is
  not implemented`. Training keeps the fast kernels; only probes switch.

Precision policy: fp16 autocast with fp32 master weights for training (T4 is
Turing and has no bf16 tensor cores), fp32 for Newton–Schulz, and fp32 with
autocast off for all curvature estimates — second derivatives under fp16 are
noise.

## Limitations

One dataset, one task, SFT only — **this does not test the RLVR regime**, since
SST-2 provides a dense signal rather than a sparse verified reward. Sharpness is
measured on a single fixed 128-example batch and only at the final step, so the
geometry evolution is visible through training but the landscape is not. One
model size. 3 seeds for the main comparison, 4 for the momentum ablation; the
sharpness and momentum effects are underpowered and are reported as such.


## References

Jordan et al. (2024), *Muon* (modded-nanogpt) · Liu et al. (2025), *Moonlight* ·
Li et al. (2018), *Visualizing the Loss Landscape of Neural Nets* · Dinh et al.
(2017), *Sharp Minima Can Generalize For Deep Nets* · Kwon et al. (2021), *ASAM*
