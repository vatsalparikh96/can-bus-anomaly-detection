# Iteration log — Phase 5

Three full-data training runs plus the original LIMIT=200k smoke run, evaluated
on the same OTIDS attack files. The point isn't just to pick a winner —
it's to capture *why* each result is what it is, so future-me (and the interviewer)
understand the hyperparameter intuition I built.

---

## Summary table

| Run | data | top_k | bottleneck | Train rows | Best val MSE | DoS PR-AUC | Fuzzy PR-AUC | Imp. PR-AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Smoke v1 | LIMIT=200k | 20 | 8 | 140k | 0.0092 | 0.868 | 0.958 | 0.877 |
| **Run 1** baseline | full | 20 | 8 | 3.15M | 0.0039 | 0.671 | 0.885 | 0.767 |
| **Run 2** wider IDs | full | 30 | 8 | 3.15M | 0.0018 | **0.694** | **0.892** | **0.797** |
| **Run 3** tight bottleneck | full | 20 | 4 | 3.15M | (n/a) | 0.627 | 0.869 | 0.760 |

Winner of the full-data sweep: **Run 2 (top_k=30, bottleneck=8)**.

---

## Run 1 — baseline on full data

**Setup:** `top_k=20`, `bottleneck=8`. Just the default config but with all
~3.15M training rows instead of the smoke run's 140k.

**What I expected:** PR-AUC equal to or better than the smoke run, because more
training data should give a more accurate model of "normal."

**What actually happened:** the model trained to a much lower MSE (val 0.0039 vs.
0.0092 for smoke — 2.4× tighter reconstruction), **but PR-AUC dropped across all
three attacks** (DoS −0.20, Fuzzy −0.07, Imp. −0.11).

**Why I think this happened.** Two effects working against each other:
- **More data → tighter "normal" distribution** in reconstruction-error space.
  The mean test error fell from 0.010 to 0.0039.
- **More data → broader coverage of legitimate driving phases**, including
  rare-but-normal frames that previously looked anomalous. The model now
  reconstructs *those* frames well too.

The net effect: the "normal" error distribution shrank, but so did the
distance between normal and attack errors. PR-AUC measures the ranking
quality of the separation — and the ranking got slightly worse, not better.

**Interview-grade takeaway:** more data is not strictly better for unsupervised
anomaly detection. Models that overfit a *narrow* sense of normal can paradoxically
be *better* anomaly detectors than models that learn a generous sense of normal —
because they call more things anomalous. The right framing is: *more data improves
modeling, not necessarily detection.*

---

## Run 2 — wider top-K (top_k=30)

**Setup:** `top_k=30`, `bottleneck=8`, full data.

**Hypothesis:** with 30 one-hot columns instead of 20, fewer normal frames fall
into the "other" bucket. The bucket gets cleaner — only rare/event-driven CAN
IDs end up there — and the model has more discriminative information per frame.

**Result:** modest but consistent improvement over Run 1:
- DoS PR-AUC: 0.671 → 0.694 (+0.023)
- Fuzzy PR-AUC: 0.885 → 0.892 (+0.007)
- Impersonation PR-AUC: 0.767 → 0.797 (+0.030)

The biggest gain is on Impersonation, which is consistent with the hypothesis:
impersonation attacks use legitimate CAN IDs, so the more those IDs are
explicitly represented in the feature vector (instead of all going to "other"),
the more the model can detect "wrong data bytes for this specific ID."

Best val MSE also dropped to 0.0018 (vs. 0.0039 at top_k=20) — more features →
better reconstruction.

**Takeaway:** widening top-K helps, especially for masquerade-style attacks.
The tradeoff is feature-vector size (40 features instead of 30), which is fine
for our tiny MLP but matters for larger models or memory-constrained edge
deployment.

---

## Run 3 — tighter bottleneck (bottleneck=4)

**Setup:** `top_k=20`, `bottleneck=4`, full data.

**Hypothesis:** a tighter bottleneck (4-dim instead of 8-dim) forces stronger
compression. Rare/anomalous frames should suffer disproportionately, sharpening
their reconstruction-error signal.

**Result:** **worse than baseline across all three attacks.**
- DoS PR-AUC: 0.671 → 0.627 (−0.044)
- Fuzzy PR-AUC: 0.885 → 0.869 (−0.016)
- Impersonation PR-AUC: 0.767 → 0.760 (−0.007)

**Why I think this happened.** Over-compression hurts both normal and attack
reconstruction equally — the bottleneck is too small to even represent normal
traffic accurately, so the "noise floor" of normal error rises. Attacks aren't
proportionally worse-reconstructed; the ranking degrades.

The hypothesis assumed normal traffic could be compressed into 4 dimensions
without major loss. In practice, the encoder needs more capacity to represent
the diversity of normal CAN frames (different ECUs sending different data
patterns), and forcing 4 dims collapses too much information.

**Takeaway:** bottleneck size is a "U-shaped" hyperparameter. Too wide and the
autoencoder is an identity function (any input reconstructs perfectly, no
discrimination). Too narrow and normal reconstruction breaks down. Default of
8 was already near the sweet spot for this feature dimensionality (~30–40).

---

## The smoke-vs-full puzzle

Why does the LIMIT=200k smoke run beat all three full-data runs?

The 200k smoke slice is **chronological** — it's the first 200k frames, which
correspond to a particular phase of the OTIDS recording (probably a single
driving condition: stationary, low-speed, or constant cruise). That makes the
"normal" class narrow: the model learns a tight, specific definition of normal,
and anything outside it — including legitimate-but-different driving — gets a
high reconstruction error.

The full-data run sees all driving phases (parking, acceleration, highway,
braking, etc.) and learns a broader definition of normal. That generalizes
better in *honest* deployment, but on the OTIDS evaluation (where attack files
are mixed with the same broad driving phases) it makes attacks look less
distinct.

**This is a real result worth discussing in interviews.** It's an example of
the bias-variance trade-off as applied to unsupervised anomaly detection — and
it shows that "more data" is not always the right answer.

A proper deployment would deal with this by:
1. Training on multiple distinct vehicle phases and reporting per-phase metrics.
2. Using sequence context (an LSTM, Phase 2.5) to capture *temporal* normality
   rather than just frame-level frequency.
3. Stratifying the test set into known driving phases to evaluate per-phase
   detection rate.

---

## Decision and promotion

**Promoting Run 2 to `results/` top-level.** Reasons:

- Best PR-AUC across all three attacks among full-data runs.
- The features (top_k=30) match the actual diversity of the OTIDS bus better
  than top_k=20.
- The bottleneck=8 size is the right midpoint between under- and over-compression
  (Run 3 demonstrated the failure mode of over-compression).

The smoke v1 numbers are kept for reference but are not the promoted result —
they're not honest in the sense that they reflect a chronological slice the
model happened to fit tightly. Full-data is the defensible number to put on
the resume.

**Final headline numbers (Run 2, OTIDS):**

| Attack | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| DoS | 0.944 | 0.047 | 0.090 | 0.694 |
| Fuzzy | 0.993 | 0.446 | 0.615 | 0.892 |
| Impersonation | 0.856 | 0.011 | 0.022 | 0.797 |

**False positive rate on attack-free test:** 0.52% (at the 99th-percentile
threshold of training reconstruction error).

---

## Phase 4.5 re-run with Run 2 winner — cross-dataset numbers

After promoting Run 2, I re-ran the Phase 4.5 cross-dataset evaluation on
CrySyS data. The smoke-run numbers shifted slightly but the story is
unchanged:

| Cross-dataset metric | Smoke v1 model | Run 2 (winner) model |
|---|---|---|
| FP rate on benign CrySyS | 40.4% | **43.6%** |
| PR-AUC pos_offset_masq | 0.049 | 0.048 |
| PR-AUC replay_inj | 0.049 | 0.048 |
| PR-AUC double_pos_inj | 0.049 | 0.052 |

The winner is somewhat *more* aggressive in flagging cross-domain data as
anomalous (43.6% FP vs. 40.4%). Hypothesis: the wider top-K (30) gives the
model more distinct CAN-ID columns to "expect" — when CrySyS frames fail to
match any of them, the model's reconstruction surprise is sharper. PR-AUC for
attack detection in the cross-domain setting stays near zero — confirming the
diagnosis that the failure is in the feature pipeline, not the model.

These are the **final cross-dataset numbers** for the resume bullet.

---

## What I'd try next (didn't run yet)

- **Wider hidden layer** (`hidden=32` instead of 16). Doubling capacity might
  let the model better represent normal diversity without losing
  discriminative power.
- **Larger top-K** (40, 50) — possibly diminishing returns; needs verification.
- **Z-score normalization** on data bytes instead of `/255` — would handle
  outlier bytes (rare but possible for sensor faults) more gracefully.
- **A "phase-stratified" evaluation** — split test set by driving phase and
  report per-phase metrics, to confirm the smoke-vs-full hypothesis from
  above quantitatively.
- **LSTM autoencoder (Phase 2.5)** — sequence-level model. Expected to lift
  Impersonation PR-AUC significantly because impersonation attacks break the
  *temporal* pattern of a CAN ID's data bytes, which the per-frame MLP cannot see.
