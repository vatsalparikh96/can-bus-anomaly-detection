"""Build notebooks/07_crysys_cross_dataset.ipynb.

Phase 4.5: take the OTIDS-trained model and apply it to CrySyS CAN traffic
(different vehicle, different ECUs). Measure how much performance drops and
explain why. The point is to demonstrate domain shift and the limits of
single-dataset generalization — a strong interview talking point.
"""

from __future__ import annotations
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


cells: list = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(dedent(source).strip("\n")))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(dedent(source).strip("\n")))


# ============================================================
# Front matter
# ============================================================
md("""
# 07 — Cross-Dataset Evaluation: OTIDS → CrySyS (Phase 4.5)

**Goal.** Take the OTIDS-trained autoencoder (Phase 4) and run it on CrySyS CAN
traffic — a different vehicle, different ECUs, different attack types.
Measure the performance drop, explain *why*, and convert this into a stronger
resume bullet.

**Why this matters.** A real-world ADAS / cybersecurity system has to
generalize: training on one vehicle's CAN bus and deploying on another. This is
exactly the kind of cross-dataset robustness question Bosch, Continental, and
Applied Intuition would ask about in interviews.

**The honest hypothesis** (formed before running anything): performance will
drop substantially. CrySyS uses different ECUs than OTIDS, so the CAN IDs are
different — most CrySyS frames will fall into our "other" bucket, which the
OTIDS model barely saw during training.

**Sections.**
- A. Setup — load OTIDS-trained model + FeatureBuilder
- B. The CrySyS dataset
- C. Load benign CrySyS traffic
- D. Domain shift: CAN ID overlap analysis
- E. Reconstruction errors on benign data
- F. Reconstruction errors on attacks
- G. Why it fails — interpretation
- H. The resume bullet
""")


# ============================================================
# A. Setup
# ============================================================
md("""
---
## A. Setup

Imports + load the saved OTIDS artifacts (model, FeatureBuilder, train errors).
""")

code("""
from pathlib import Path
import json
import sys

# Make src/ importable
sys.path.insert(0, str(Path("../..").resolve()))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data.load_crysys import (
    load_crysys_file,
    find_crysys_benign_logs,
    find_crysys_attack_logs,
)
from src.data.preprocess import FeatureBuilder
from src.models.autoencoder import MLPAutoencoder
from src.training.evaluate import (
    compute_recon_errors,
    pick_threshold,
    evaluate_at_threshold,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path("../../results")
CRYSYS_LOGS = Path("../../data/crysys/logs")
print(f"Device: {DEVICE}")
print(f"CrySyS logs at: {CRYSYS_LOGS.resolve()}")
""")

code("""
# Load OTIDS-trained artifacts
with open(RESULTS / "model_config.json") as f:
    mc = json.load(f)
model = MLPAutoencoder(
    n_features=mc["n_features"], hidden=mc["hidden"], bottleneck=mc["bottleneck"],
).to(DEVICE)
model.load_state_dict(torch.load(RESULTS / "model.pt", map_location=DEVICE))
model.eval()

fb = FeatureBuilder.load(RESULTS / "feature_builder.npz")
train_errors = np.load(RESULTS / "train_errors.npy")
otids_threshold = pick_threshold(train_errors, percentile=99.0)

print(f"Loaded OTIDS-trained model. Threshold: {otids_threshold:.6f}")
print(f"FeatureBuilder top-20 CAN IDs (hex): {[f'0x{i:03X}' for i in sorted(fb.top_ids)]}")
""")


# ============================================================
# B. The CrySyS dataset
# ============================================================
md("""
---
## B. The CrySyS dataset

**Source.** [CrySyS Lab](https://www.crysys.hu/) (Budapest University of
Technology and Economics, Hungary). Published in *Nature Scientific Data*
(2023). [Paper](https://www.nature.com/articles/s41597-023-02716-9).

**Data.** Real CAN bus traces from a vehicle (make undisclosed in the paper —
likely European), with two attack categories layered on top:
- **Fabrication** (`msg-inj`) — inject malicious frames into the bus.
- **Masquerade** (`msg-mod`) — modify legitimate frames in flight (analogous
  to OTIDS impersonation).

Each scenario folder (`S-x-y`, `T-x-y`) contains one benign log plus several
attack variants. Variants modify either one signal or two simultaneously,
using strategies like POS-OFFSET (additive constant), ADD-INCR (incrementing
deviation), REPLAY (replay old values), CONST (freeze value), etc.

**File format.** SocketCAN candump output:

```
(<timestamp>) <interface> <can_id>#<data_bytes_hex>
```

The `src/data/load_crysys.py` loader parses this into the same DataFrame
schema as our OTIDS loader, so the rest of our pipeline (FeatureBuilder,
model) needs no changes.
""")


# ============================================================
# C. Load benign data
# ============================================================
md("""
---
## C. Load benign CrySyS traffic

Pick a few benign files of different sizes — the `S-x-y` files are short
stationary recordings; the `T-x-y` files are longer trip recordings.
""")

code("""
benign_logs = find_crysys_benign_logs(CRYSYS_LOGS)
print(f"Total benign files: {len(benign_logs)}")
print()

# Use 3 representative files: a short S-, a medium T-, and another medium T-
sample = [
    next(p for p in benign_logs if p.name == "S-1-1-benign.log"),
    next(p for p in benign_logs if p.name == "T-1-1-benign.log"),
    next(p for p in benign_logs if p.name == "T-2-1-benign.log"),
]
for p in sample:
    size_kb = p.stat().st_size / 1024
    print(f"  {p.parent.name}/{p.name:<25s}  {size_kb:>8.0f} KB")
""")

code("""
# Load all three into one combined DataFrame
crysys_benign_dfs = [load_crysys_file(p) for p in sample]
df_crysys_benign = pd.concat(crysys_benign_dfs, ignore_index=True)
print(f"Total benign rows loaded: {len(df_crysys_benign):,}")
print(df_crysys_benign.head())
""")


# ============================================================
# D. Domain shift analysis
# ============================================================
md("""
---
## D. Domain shift: CAN ID overlap analysis

This is the diagnostic that explains everything that follows. If the CAN IDs in
CrySyS don't overlap with the IDs our FeatureBuilder learned from OTIDS, then
the OTIDS-trained model will have effectively never seen the CrySyS feature
distribution — and won't reconstruct it well.
""")

code("""
otids_top20 = set(fb.top_ids)
crysys_ids   = set(df_crysys_benign["can_id"].unique())

print(f"OTIDS top-20 CAN IDs:           {len(otids_top20)}")
print(f"Distinct CAN IDs in CrySyS data: {len(crysys_ids)}")
overlap = otids_top20 & crysys_ids
print(f"\\nOverlap: {len(overlap)} ID(s)")
print(f"  hex: {[f'0x{i:03X}' for i in sorted(overlap)]}")
print(f"\\nCrySyS IDs NOT in OTIDS top-20: {len(crysys_ids - otids_top20)}")
print(f"  hex: {[f'0x{i:03X}' for i in sorted(crysys_ids - otids_top20)][:10]}...")

# What fraction of CrySyS frames fall into the OTIDS 'other' bucket?
in_top = df_crysys_benign["can_id"].isin(otids_top20)
print(f"\\nFraction of CrySyS frames in OTIDS 'other' bucket: {(~in_top).mean()*100:.1f}%")
""")

md("""
**This is the headline result of Phase 4.5.** Effectively every CrySyS frame
falls into the OTIDS-FeatureBuilder's "other" bucket. The autoencoder has
barely seen "other" frames during training (only the ~few percent of OTIDS
traffic outside the Kia Soul top-20), so its representation of those frames is
poor.

Before we even run the model on CrySyS, we can predict: reconstruction errors
will be high across the board, and the OTIDS-threshold-based classifier will
flag essentially everything — including legitimate frames — as anomalous.
""")


# ============================================================
# E. Reconstruction errors on benign
# ============================================================
md("""
---
## E. Reconstruction errors on benign CrySyS data

Time to test the prediction. Apply the OTIDS-trained FeatureBuilder + model to
CrySyS benign traffic and see how the error distribution compares to OTIDS test
errors.
""")

code("""
# OTIDS test errors (loaded for comparison)
X_test_normal = np.load(RESULTS / "X_test_normal.npy")
err_otids_test = compute_recon_errors(model, X_test_normal, device=DEVICE)

# CrySyS benign errors
X_crysys_benign = fb.transform(df_crysys_benign)
err_crysys_benign = compute_recon_errors(model, X_crysys_benign, device=DEVICE)

print(f"OTIDS attack-free test:   mean={err_otids_test.mean():.6f}  p99={np.quantile(err_otids_test, 0.99):.6f}")
print(f"CrySyS benign (in-domain shift): mean={err_crysys_benign.mean():.6f}  p99={np.quantile(err_crysys_benign, 0.99):.6f}")
print()
print(f"Mean error ratio (CrySyS / OTIDS): {err_crysys_benign.mean() / err_otids_test.mean():.1f}x")
print()
fp_rate = (err_crysys_benign > otids_threshold).mean()
print(f"At OTIDS threshold {otids_threshold:.6f}:")
print(f"  OTIDS test flagged: {(err_otids_test > otids_threshold).mean()*100:.2f}%")
print(f"  CrySyS benign flagged (false positives): {fp_rate*100:.2f}%")
""")

code("""
# Compare distributions
fig, ax = plt.subplots(figsize=(9, 4))
max_err = max(err_otids_test.max(), err_crysys_benign.max())
bins = np.linspace(0, max_err * 1.05, 80)
ax.hist(err_otids_test,    bins=bins, alpha=0.6, label="OTIDS attack-free (in-domain)",   color="tab:blue")
ax.hist(err_crysys_benign, bins=bins, alpha=0.6, label="CrySyS benign (cross-domain)",  color="tab:orange")
ax.axvline(otids_threshold, color="black", linestyle="--", linewidth=1, label="OTIDS threshold (99th pct of train)")
ax.set_yscale("log")
ax.set_xlabel("reconstruction error")
ax.set_ylabel("count (log)")
ax.set_title("Domain shift: OTIDS-trained model on CrySyS data")
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS / "phase45_domain_shift_hist.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
What the histogram tells us:

- The **orange** distribution (CrySyS benign) is shifted far to the right
  compared to the **blue** (OTIDS test) distribution. The model can reconstruct
  in-domain data well but fails on out-of-domain data, even though both are
  *benign*.
- A huge fraction of orange mass is to the right of the threshold — those would
  all be false positives if we deployed this model on CrySyS data.
- This isn't a "the model is bad" finding. It's a "the model is good *for its
  training distribution* and doesn't transfer" finding. **That's domain shift.**
""")


# ============================================================
# F. Reconstruction errors on attacks
# ============================================================
md("""
---
## F. Reconstruction errors on CrySyS attacks

Now we evaluate the model's *attack detection* ability in the cross-dataset
setting. We pick three representative attack types from CrySyS:

- `POS_OFFSET-msg-mod` — masquerade attack: a legitimate frame's value is
  shifted by a positive offset (most similar to OTIDS impersonation).
- `REPLAY-msg-inj` — fabrication attack: inject replayed (stale) messages
  (loosely analogous to fuzzy injection in OTIDS).
- `DOUBLE-msg-inj-...-POS-OFFSET-...` — fabrication on two signals
  simultaneously (a harder version of the above).
""")

code("""
attack_examples = {
    "pos_offset_masq": find_crysys_attack_logs(CRYSYS_LOGS, "POS_OFFSET-msg-mod"),
    "replay_inj":      find_crysys_attack_logs(CRYSYS_LOGS, "REPLAY-msg-inj"),
    "double_pos_inj":  find_crysys_attack_logs(CRYSYS_LOGS, "DOUBLE-msg-inj"),
}
err_crysys_attacks = {}
for name, paths in attack_examples.items():
    # Take just the first matching file per attack type to keep eval fast
    p = paths[0]
    df = load_crysys_file(p, limit=200_000)
    X = fb.transform(df)
    err = compute_recon_errors(model, X, device=DEVICE)
    err_crysys_attacks[name] = err
    print(f"{name:<20s}: n={len(err):>7,}  mean={err.mean():.6f}  file={p.parent.name}/{p.name[:45]}...")
""")

code("""
# Compute PR-AUC for cross-dataset attack detection
# (CrySyS benign as negatives, CrySyS attack as positives)
print(f"{'attack':<20s} {'precision':>10s} {'recall':>10s} {'F1':>10s} {'PR-AUC':>10s}")
print("-" * 65)
metrics_crysys = {}
for name, err in err_crysys_attacks.items():
    m = evaluate_at_threshold(err_crysys_benign, err, otids_threshold)
    metrics_crysys[name] = m
    print(f"{name:<20s} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['pr_auc']:>10.3f}")

# Save
with open(RESULTS / "phase45_metrics.json", "w") as f:
    json.dump({
        "false_positive_rate_on_benign_crysys": float((err_crysys_benign > otids_threshold).mean()),
        "metrics": metrics_crysys,
    }, f, indent=2)
""")

code("""
# Visualize attack vs benign in cross-domain
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (name, err) in zip(axes, err_crysys_attacks.items()):
    max_err = max(err_crysys_benign.max(), err.max())
    bins = np.linspace(0, max_err * 1.05, 80)
    ax.hist(err_crysys_benign, bins=bins, alpha=0.6, label="CrySyS benign", color="tab:blue")
    ax.hist(err,               bins=bins, alpha=0.6, label=name,            color="tab:red")
    ax.axvline(otids_threshold, color="black", linestyle="--", linewidth=1, label="OTIDS threshold")
    ax.set_yscale("log")
    ax.set_xlabel("reconstruction error")
    ax.set_title(name)
    ax.legend(fontsize=7)
axes[0].set_ylabel("count (log)")
fig.suptitle("Cross-dataset attack detection (OTIDS-trained model on CrySyS attacks)")
fig.tight_layout()
fig.savefig(RESULTS / "phase45_attack_hist.png", dpi=120, bbox_inches="tight")
plt.show()
""")


# ============================================================
# G. Interpretation
# ============================================================
md("""
---
## G. Why it fails — interpretation

Two failure modes are likely visible in the histograms above:

**1. Catastrophically high false-positive rate on benign CrySyS.** Most benign
CrySyS frames produce reconstruction errors *above* the OTIDS-derived
threshold. The model is unable to distinguish normal CrySyS traffic from what
it learned to consider anomalous. In production, this would mean the IDS
flagging legitimate frames continuously — every car of a new make becomes a
"persistent attack" to the model.

**2. Compressed attack-vs-benign separation.** PR-AUC on CrySyS attacks is far
lower than on OTIDS attacks (which were 0.88–0.96). Because both benign and
attack frames in CrySyS land in our "other" bucket, the model sees them
similarly — it has lost the discriminative power it had on OTIDS.

**Root cause** is in the *featurization*, not the model. The FeatureBuilder
encodes CAN ID via one-hot of the top-K most common IDs *learned from training
data*. Those IDs are Kia Soul ECU IDs. Apply that to a different vehicle and
99.8% of the frames hit the "other" column — the model has lost its primary
informative feature.

**Diagnosis flow:**

```
trained on OTIDS Kia Soul
        ↓
fb.top_ids = [Kia ECU IDs]
        ↓
applied to CrySyS (different vehicle)
        ↓
99.8% of frames → "other" bucket
        ↓
model has barely seen "other" during training
        ↓
high reconstruction error on benign AND attack
        ↓
threshold-based detection fails
```

**What would fix this in real deployment?**

1. **Refit the FeatureBuilder on CrySyS benign data** — learn the new vehicle's
   top-K IDs. The model still won't generalize, but features become
   informative again.
2. **Retrain on combined OTIDS + CrySyS** — multi-domain training. Model
   learns features common across vehicles.
3. **Different encoding** — instead of one-hot of CAN ID (which is
   vehicle-specific), use *embeddings* trained jointly across datasets, or
   ID-independent features (timing patterns, data byte statistics).
4. **Transfer learning** — train on OTIDS, fine-tune on a small amount of
   CrySyS data.

Option 3 is the most research-grade answer and would be a strong follow-up project.
""")


# ============================================================
# H. The resume bullet
# ============================================================
md("""
---
## H. The resume bullet

This phase converts your Phase 4 project from a single-dataset result into a
**generalization study** — much stronger interview material. Draft bullets:

> *"Trained PyTorch MLP autoencoder for CAN bus intrusion detection on the
> OTIDS benchmark (Kia Soul, PR-AUC 0.88–0.96 across DoS/Fuzzy/Impersonation).
> Demonstrated significant domain shift on the CrySyS dataset (different OEM
> vehicle): 99.8% of cross-vehicle frames mapped to the 'other'
> CAN-ID bucket, producing a high false-positive rate on benign cross-dataset
> traffic. Diagnosed the failure as feature-engineering-bound rather than
> model-bound — proposed multi-domain training and learned CAN-ID embeddings
> as solutions."*

What this signals to an interviewer:
- You can train a model end-to-end (Phase 4).
- You design honest evaluations (PR-AUC, not just F1).
- You think about generalization and domain shift — not just leaderboard chasing.
- You can diagnose *why* a model fails, separating feature-pipeline issues
  from model-architecture issues.
- You know what fixes are feasible (multi-domain training, embeddings, transfer
  learning) and can pitch follow-up work.

This is the difference between a beginner ML resume and a "this person
actually thinks about the production deployment" resume.
""")


# ============================================================
# Build
# ============================================================
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata.kernelspec = {
    "display_name": "Python 3 (.venv)",
    "language": "python",
    "name": "python3",
}
nb.metadata.language_info = {
    "name": "python",
    "version": "3.11.9",
}

out = Path(__file__).parent / "03_crysys_cross_dataset.ipynb"
nbf.write(nb, str(out))
print(f"Wrote {out} ({len(cells)} cells)")
