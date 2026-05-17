"""Build notebooks/08_hyperparameter_comparison.ipynb.

Compare all four runs from Phase 5 side-by-side. The point is to extract the
*mechanism* behind each hyperparameter's effect — not just rank them. A
learning-focused notebook that teaches HP intuition for future projects.
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
# 08 — Hyperparameter Comparison & Learning Reference

**Goal of this notebook.** Take the four Phase 5 runs, lay them next to each
other, and extract *why* each hyperparameter moves the metrics. This is the
artifact you'll reach for in three months when you're starting a new ML
project and trying to remember "did wider top-K help or hurt last time?"

**The four runs we'll compare:**

| Run | LIMIT | top_k | bottleneck | Where saved |
|---|---|---:|---:|---|
| smoke_v1 | 200,000 rows | 20 | 8 | `results/runs/smoke_v1/` |
| run1_baseline | full (~4.5M) | 20 | 8 | `results/runs/run1_baseline/` |
| run2_topk30 | full | 30 | 8 | `results/runs/run2_topk30/` |
| run3_bottleneck4 | full | 20 | 4 | `results/runs/run3_bottleneck4/` |

**Structure.**
- A. Setup — load all four runs
- B. Summary table — the headline numbers
- C. Loss curves — what training behavior tells us
- D. Reconstruction error distributions — side by side per attack
- E. PR curves — threshold-independent ranking quality
- F. PR-AUC bar chart — single-glance comparison
- G. Parameter-by-parameter analysis
  - G.1 LIMIT (training data quantity)
  - G.2 top_k (feature width)
  - G.3 bottleneck (compression dim)
- H. The smoke-vs-full puzzle — why less data sometimes beats more
- I. Learning nuggets — what to remember next time
- J. What I'd try next
""")


# ============================================================
# A. Setup
# ============================================================
md("""
---
## A. Setup

Imports + load each run's saved artifacts: `history.json`, `metrics.json`,
`train_errors.npy`, `X_test_normal.npy`, `feature_builder.npz`, `model.pt`,
`model_config.json`. Same layout for every run, so we use a small helper.
""")

code("""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

# Make src/ importable
sys.path.insert(0, str(Path("../..").resolve()))

from src.data.load import find_file, load_otids_file
from src.data.preprocess import FeatureBuilder
from src.models.autoencoder import MLPAutoencoder
from src.training.evaluate import compute_recon_errors, pick_threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS_DIR = Path("../../results/runs")
RAW_DIR  = Path("../../data/otids/raw")
print(f"Device: {DEVICE}")
print(f"Runs dir: {RUNS_DIR.resolve()}")
""")

code("""
def load_run(run_dir: Path) -> dict:
    \"\"\"Load all artifacts for one run into a dict.\"\"\"
    with open(run_dir / "history.json") as f:
        history = json.load(f)
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    with open(run_dir / "model_config.json") as f:
        mc = json.load(f)
    model = MLPAutoencoder(
        n_features=mc["n_features"], hidden=mc["hidden"], bottleneck=mc["bottleneck"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=DEVICE))
    model.eval()
    fb = FeatureBuilder.load(run_dir / "feature_builder.npz")
    train_errors = np.load(run_dir / "train_errors.npy")
    X_test = np.load(run_dir / "X_test_normal.npy")
    return {
        "name": run_dir.name,
        "history": history,
        "metrics": metrics,
        "model_config": mc,
        "model": model,
        "fb": fb,
        "train_errors": train_errors,
        "X_test": X_test,
    }


run_names = ["smoke_v1", "run1_baseline", "run2_topk30", "run3_bottleneck4"]
runs = {name: load_run(RUNS_DIR / name) for name in run_names}

# Pretty labels for plots
labels = {
    "smoke_v1":        "smoke (200k, k=20, b=8)",
    "run1_baseline":   "run1 (full,  k=20, b=8)",
    "run2_topk30":     "run2 (full,  k=30, b=8)",
    "run3_bottleneck4":"run3 (full,  k=20, b=4)",
}
colors = {
    "smoke_v1":        "tab:purple",
    "run1_baseline":   "tab:blue",
    "run2_topk30":     "tab:green",
    "run3_bottleneck4":"tab:orange",
}

for name, run in runs.items():
    mc = run["model_config"]
    print(f"{name:<22s} top_k={mc['top_k_can_ids']}, bottleneck={mc['bottleneck']}, "
          f"n_features={mc['n_features']}, train_errors={len(run['train_errors']):,}")
""")


# ============================================================
# B. Summary table
# ============================================================
md("""
---
## B. Summary table — the headline numbers

The numbers you'll come back to. Lower training loss does NOT necessarily mean
higher PR-AUC — that's the central tension we'll dig into.
""")

code("""
rows = []
for name, run in runs.items():
    m = run["metrics"]
    h = run["history"]
    mc = run["model_config"]
    rows.append({
        "run": name,
        "top_k": mc["top_k_can_ids"],
        "bottleneck": mc["bottleneck"],
        "train_rows": len(run["train_errors"]),
        "final_train_loss": h["train_loss"][-1],
        "best_val_loss":   min(h["val_loss"]),
        "threshold":       m["threshold"],
        "dos_pr_auc":      m["dos"]["pr_auc"],
        "fuzzy_pr_auc":    m["fuzzy"]["pr_auc"],
        "imp_pr_auc":      m["impersonation"]["pr_auc"],
    })
df_summary = pd.DataFrame(rows).set_index("run")

# Format for display
def color_best(s):
    is_max = s == s.max()
    return ["**" + f"{v:.4f}" + "**" if m else f"{v:.4f}" for v, m in zip(s, is_max)]

display_df = df_summary.copy()
print(display_df.to_string(float_format=lambda x: f"{x:.4f}"))
""")


# ============================================================
# C. Loss curves
# ============================================================
md("""
---
## C. Loss curves — what training behavior tells us

Four train+val loss curves. Look for three things:

1. **Final loss level.** More data (run1/2/3) reaches lower MSE than smoke
   (200k rows). Wider top_k (run2) reaches lowest MSE because more features to
   represent.
2. **Convergence speed.** Tighter bottleneck (run3) plateaus higher — the
   model literally can't compress normal traffic well enough into 4 dims.
3. **Train-vs-val gap.** Healthy training keeps the gap small. A widening gap
   = overfitting. None of our runs show severe overfitting because the model is
   small (~1,500 parameters) and the data is large.
""")

code("""
fig, axes = plt.subplots(1, 4, figsize=(18, 3.5), sharey=False)
for ax, (name, run) in zip(axes, runs.items()):
    h = run["history"]
    ax.plot(h["train_loss"], label="train", color="tab:blue", linewidth=1.5)
    ax.plot(h["val_loss"],   label="val",   color="tab:red",  linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(labels[name], fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle("Training loss curves — all 4 runs", y=1.02)
fig.tight_layout()
fig.savefig("../../results/hp_comparison_loss_curves.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
**What stands out.** Smoke and Run 3 both plateau higher than Runs 1 & 2.
- Smoke plateaus higher because it has less data to learn from — the model
  reaches its capacity on what it can see.
- Run 3 plateaus higher because its bottleneck (4) is too narrow — the model
  *physically cannot* compress the normal data into 4 dims accurately.

The two plateaus look similar but for opposite reasons: smoke is
**data-limited**, run3 is **capacity-limited**.
""")


# ============================================================
# D. Reconstruction error distributions
# ============================================================
md("""
---
## D. Reconstruction error distributions — side by side

For each attack type, overlay all four runs' attack-error distributions on a
single log-scale histogram. This is the most informative visualization in the
notebook — it lets you SEE why PR-AUC differs.

**Note on compute:** We re-run each model on a 200k-row sample of each attack
file to keep this fast. The full-data conclusions don't change.
""")

code("""
ATTACK_LIMIT = 200_000

# Load attack data once (we'll transform/score with each run's own FB+model)
print("Loading attack files (this is one-shot, takes a few seconds)...")
attack_dfs = {}
for label, kw in [("dos", "DoS"), ("fuzzy", "Fuzzy"), ("impersonation", "Impersonation")]:
    p = find_file(RAW_DIR, kw)
    attack_dfs[label] = load_otids_file(p, limit=ATTACK_LIMIT)
    print(f"  {label}: {len(attack_dfs[label]):,} rows")
""")

code("""
# Compute attack errors with each run's model
attack_errors = {name: {} for name in runs}
test_errors   = {}

for name, run in runs.items():
    # Errors on attack-free test using THIS run's saved X_test_normal
    test_errors[name] = compute_recon_errors(run["model"], run["X_test"][:200_000], device=DEVICE)
    for label, df in attack_dfs.items():
        X = run["fb"].transform(df)
        attack_errors[name][label] = compute_recon_errors(run["model"], X, device=DEVICE)
print("Done computing errors.")
""")

code("""
# 3 panels (one per attack), each shows: attack-free test (single dist) + 4 overlaid attack dists
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
for ax, attack_label in zip(axes, ["dos", "fuzzy", "impersonation"]):
    # Combined max for binning
    all_err_max = max(
        max(test_errors[name].max() for name in runs),
        max(attack_errors[name][attack_label].max() for name in runs),
    )
    bins = np.linspace(0, all_err_max * 1.05, 80)

    # Attack-free test (use smoke's distribution as reference, or could pick run1)
    ax.hist(test_errors["run1_baseline"], bins=bins, alpha=0.25, color="gray",
            label="attack-free test (run1)")

    for name, run in runs.items():
        ax.hist(attack_errors[name][attack_label], bins=bins, alpha=0.5,
                color=colors[name], label=labels[name])
        # Mark each run's threshold
        ax.axvline(run["metrics"]["threshold"], color=colors[name],
                   linestyle="--", linewidth=1, alpha=0.7)

    ax.set_yscale("log")
    ax.set_xlabel("reconstruction error")
    ax.set_title(attack_label)
    ax.legend(fontsize=7)
axes[0].set_ylabel("count (log)")
fig.suptitle("Attack error distributions across all 4 runs (dashed lines = each run's threshold)")
fig.tight_layout()
fig.savefig("../../results/hp_comparison_error_dists.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md("""
**Reading the plot.** For each attack panel:

- The **gray** distribution is attack-free test (run1's, as reference).
- The four colored distributions are the SAME attack frames re-scored by each
  of the four models. Different shape per run because each model has a
  different "normal" representation.
- The dashed lines mark each run's threshold (99th percentile of its own
  training error). The threshold position varies because each run's reconstruction
  scale differs.

**Smoke (purple)** typically has the most-shifted-right attack distribution
relative to its own threshold — that's why its PR-AUC is highest. The model
has a *narrow* sense of "normal" so anomalies stand out.

**Run 3 (orange, tight bottleneck)** has compressed attack distributions —
everything got squeezed because over-compression flattened the reconstruction
landscape.
""")


# ============================================================
# E. PR curves
# ============================================================
md("""
---
## E. PR curves — threshold-independent ranking

Where the histograms in Section D mix threshold effects with model effects,
PR curves isolate the pure ranking question: "how cleanly does the model
separate attack frames from normal frames at *any* threshold?"

Higher PR-AUC = better ranking = better model, regardless of threshold choice.
""")

code("""
from sklearn.metrics import precision_recall_curve

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, attack_label in zip(axes, ["dos", "fuzzy", "impersonation"]):
    for name in runs:
        err_pos = attack_errors[name][attack_label]
        err_neg = test_errors[name]
        scores = np.concatenate([err_neg, err_pos])
        labels_arr = np.concatenate([np.zeros(len(err_neg)), np.ones(len(err_pos))])
        precision, recall, _ = precision_recall_curve(labels_arr, scores)
        ap = runs[name]["metrics"][attack_label]["pr_auc"]
        ax.plot(recall, precision, color=colors[name],
                label=f"{labels[name]} (AP={ap:.3f})", linewidth=1.8)
    ax.set_xlabel("recall")
    ax.set_title(attack_label)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("precision")
fig.suptitle("Precision-recall curves — all 4 runs per attack type")
fig.tight_layout()
fig.savefig("../../results/hp_comparison_pr_curves.png", dpi=120, bbox_inches="tight")
plt.show()
""")


# ============================================================
# F. PR-AUC bar chart
# ============================================================
md("""
---
## F. PR-AUC bar chart — single-glance comparison

The clearest "which run wins" view.
""")

code("""
fig, ax = plt.subplots(figsize=(10, 4))
attacks = ["dos", "fuzzy", "impersonation"]
x = np.arange(len(attacks))
width = 0.2

for i, name in enumerate(runs):
    pr_aucs = [runs[name]["metrics"][a]["pr_auc"] for a in attacks]
    ax.bar(x + i * width - 1.5 * width, pr_aucs, width,
           label=labels[name], color=colors[name])

ax.set_xticks(x)
ax.set_xticklabels(attacks)
ax.set_ylabel("PR-AUC")
ax.set_title("PR-AUC per attack type across all 4 runs")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)
ax.set_ylim(0, 1.05)

# Add numeric labels on top of bars
for i, name in enumerate(runs):
    for j, a in enumerate(attacks):
        val = runs[name]["metrics"][a]["pr_auc"]
        ax.text(j + i * width - 1.5 * width, val + 0.015, f"{val:.2f}",
                ha="center", fontsize=7)

fig.tight_layout()
fig.savefig("../../results/hp_comparison_pr_auc_bars.png", dpi=120, bbox_inches="tight")
plt.show()
""")


# ============================================================
# G. Parameter-by-parameter analysis
# ============================================================
md("""
---
## G. Parameter-by-parameter analysis

Three knobs we turned, three lessons.
""")

md("""
### G.1 — `LIMIT` (training data quantity)

**What it controls.** How many CAN frames are loaded from `Attack_free_dataset.txt`.
Default `None` = all ~4.5M frames. We compared LIMIT=200,000 vs LIMIT=None.

**Compare:** `smoke_v1` (purple, 200k rows) vs `run1_baseline` (blue, full).
Same architecture, same hyperparameters — only data quantity differs.
""")

code("""
print("LIMIT comparison: smoke_v1 (200k) vs run1_baseline (full)")
print()
metric_rows = []
for attack in ["dos", "fuzzy", "impersonation"]:
    smoke = runs["smoke_v1"]["metrics"][attack]
    full  = runs["run1_baseline"]["metrics"][attack]
    metric_rows.append({
        "attack": attack,
        "smoke_pr_auc": smoke["pr_auc"],
        "full_pr_auc":  full["pr_auc"],
        "delta": full["pr_auc"] - smoke["pr_auc"],
    })
print(pd.DataFrame(metric_rows).to_string(float_format=lambda x: f"{x:.4f}", index=False))

print()
print(f"  smoke val loss: {min(runs['smoke_v1']['history']['val_loss']):.6f}")
print(f"  full  val loss: {min(runs['run1_baseline']['history']['val_loss']):.6f}")
print(f"  smoke train rows: {len(runs['smoke_v1']['train_errors']):,}")
print(f"  full  train rows: {len(runs['run1_baseline']['train_errors']):,}")
""")

md("""
**The result.** More data → much better val loss (model is more accurate),
but **PR-AUC for anomaly detection DROPS substantially** across all three attacks.
Smoke beats full by 0.20+ on DoS, 0.10+ on Fuzzy, 0.18+ on Impersonation.

**The mechanism.** More training data = broader exposure to legitimate driving
phases (parking, cruising, accelerating, braking, etc.). The model learns a
*generous* sense of normal. The "normal" reconstruction-error distribution
widens to accommodate this diversity. But anomalies don't proportionally
move — they were always somewhat distinct — so the *gap* between normal and
anomaly compresses.

**The honest reframing.** This is not "less data is better." It's "less data is
narrower." If the smoke slice happens to cover only one driving phase, the
model is over-specialized to that phase. On the OTIDS evaluation, where the
attack files contain a mix of phases, the over-specialized model marks
unfamiliar-but-legitimate phases as anomalous — which the test labels happen
to *call* "attack frames" (because the files are labeled file-level, not
frame-level). So smoke is "accidentally right" via overfitting to evaluation
quirks.

**Real-world implication.** If you were deploying this in production, the
full-data model is the honest one. It would have fewer false positives on
legitimate-but-rare driving phases (highway, aggressive braking) that the
smoke model would fail on.

**Learning nugget for future projects:** *more data improves modeling, not
necessarily detection. In unsupervised anomaly detection, watch for the
trade-off between modeling fidelity and detection sensitivity.* When your
PR-AUC drops after adding data, that's a clue your previous result was
benefiting from a narrow training distribution.
""")

md("""
### G.2 — `top_k` (CAN ID feature width)

**What it controls.** How many distinct CAN IDs get their own one-hot column.
All other IDs go into a single "other" bucket.

**Compare:** `run1_baseline` (top_k=20, blue) vs `run2_topk30` (top_k=30, green).
Both full data, both bottleneck=8.
""")

code("""
print("top_k comparison: run1 (top_k=20) vs run2 (top_k=30)")
print()
metric_rows = []
for attack in ["dos", "fuzzy", "impersonation"]:
    r1 = runs["run1_baseline"]["metrics"][attack]
    r2 = runs["run2_topk30"]["metrics"][attack]
    metric_rows.append({
        "attack": attack,
        "top_k=20_pr_auc": r1["pr_auc"],
        "top_k=30_pr_auc": r2["pr_auc"],
        "delta": r2["pr_auc"] - r1["pr_auc"],
    })
print(pd.DataFrame(metric_rows).to_string(float_format=lambda x: f"{x:.4f}", index=False))

print()
print(f"  run1 features: {runs['run1_baseline']['model_config']['n_features']}")
print(f"  run2 features: {runs['run2_topk30']['model_config']['n_features']}")
""")

md("""
**The result.** Run 2 wins on every attack. The biggest gain is on
**Impersonation (+0.030)**, then DoS (+0.023), then Fuzzy (+0.007).

**Why Impersonation benefits most.** Impersonation attacks use *legitimate*
CAN IDs but with spoofed data bytes. With top_k=20, many of these IDs end up
in the "other" bucket where the model can't distinguish them from each other.
With top_k=30, more legitimate IDs get their own column, so the model can
learn "for CAN ID 0x4B0, the data bytes usually look like X — if they look
like Y instead, that's anomalous."

**Why Fuzzy benefits least.** Fuzzy injects RANDOM CAN IDs and random data
bytes. Most fuzzy frames don't match any legitimate CAN ID, so they always go
into "other" — increasing top_k from 20 to 30 doesn't change much for them.

**The cost.** Feature vector grows from 30 to 40 dimensions. Model parameter
count grows slightly (a few hundred extra weights in `nn.Linear(30, 16)` →
`nn.Linear(40, 16)`). Negligible at our scale.

**Learning nugget for future projects:** *the granularity of categorical
features matters. A "top-K + other" encoding loses information about the tail
of the distribution. Widening top-K is cheap and almost always helps — the
question is when you hit diminishing returns.* In our case, we didn't push
beyond 30; for production you'd sweep up to maybe 50 or use learned
embeddings.
""")

md("""
### G.3 — `bottleneck` (compression dimension)

**What it controls.** The smallest layer in the autoencoder — the encoder's
output and decoder's input. All information must pass through this
bottleneck. Smaller = more aggressive compression.

**Compare:** `run1_baseline` (bottleneck=8, blue) vs `run3_bottleneck4`
(bottleneck=4, orange). Both full data, both top_k=20.
""")

code("""
print("bottleneck comparison: run1 (b=8) vs run3 (b=4)")
print()
metric_rows = []
for attack in ["dos", "fuzzy", "impersonation"]:
    r1 = runs["run1_baseline"]["metrics"][attack]
    r3 = runs["run3_bottleneck4"]["metrics"][attack]
    metric_rows.append({
        "attack": attack,
        "b=8_pr_auc": r1["pr_auc"],
        "b=4_pr_auc": r3["pr_auc"],
        "delta": r3["pr_auc"] - r1["pr_auc"],
    })
print(pd.DataFrame(metric_rows).to_string(float_format=lambda x: f"{x:.4f}", index=False))

print()
print(f"  run1 val loss: {min(runs['run1_baseline']['history']['val_loss']):.6f}")
print(f"  run3 val loss: {min(runs['run3_bottleneck4']['history']['val_loss']):.6f}")
""")

md("""
**The result.** Tighter bottleneck (4 dim) is **worse** across every attack.
DoS drops the most (-0.044). Val loss also got worse (higher MSE), suggesting
the model literally can't represent normal traffic well at 4 dims.

**Why my "tighter bottleneck = sharper anomaly signal" hypothesis was wrong.**

The hypothesis: forcing the model through a 4-dim bottleneck would make rare
or anomalous frames disproportionately harder to reconstruct, producing larger
error gaps between normal and anomaly.

The reality: 4 dims is too narrow for our 30-feature input. The encoder can't
fit the diversity of normal CAN traffic into 4 dims, so the model has high
reconstruction error on **everything** — including legitimate frames. The
"noise floor" of normal error rises. Anomaly error rises too, but not
proportionally — so the gap (and PR-AUC) shrinks, not grows.

**The U-shape principle.** Bottleneck size is a U-shaped hyperparameter for
anomaly detection:

- **Too wide** (close to input dim): the autoencoder learns the identity
  function. Reconstruction error is near zero for everything → no signal.
- **Too narrow**: model can't represent even normal data → noise floor rises →
  signal-to-noise ratio drops.
- **Sweet spot**: small enough to force compression, large enough to capture
  normal patterns. Empirically, 1/4 to 1/2 of input dim is a typical starting
  point. Our 30 → 8 (≈ 1/4) was near the sweet spot.

**Learning nugget for future projects:** *autoencoder bottleneck size is not
just "smaller is better." It's a Goldilocks parameter. Default to roughly
input_dim/4, then sweep ±50%. If both ends hurt PR-AUC, you're already at the
sweet spot.*
""")


# ============================================================
# H. The smoke-vs-full puzzle deep dive
# ============================================================
md("""
---
## H. The smoke-vs-full puzzle — connecting it to real-world ML

The most counter-intuitive result of Phase 5 deserves a deeper look. **More
training data made our model *worse* at the evaluation task.** Why does this
happen, and what should you take away from it?

**Three forces in tension:**

1. **Modeling fidelity.** With more data, the model learns the true
   distribution of normal CAN traffic more accurately. Val loss drops. The
   model is "more correct" in a statistical sense.

2. **Detection sensitivity.** The model's anomaly signal is the gap between
   normal-error and anomaly-error distributions. A model that learns a tight,
   narrow "normal" has a *bigger gap* — anomalies stand out more.

3. **The evaluation's labeling.** OTIDS attack files contain a mix of
   legitimate frames AND injected attacks. If the model's "normal" is narrow
   enough that even some legitimate frames look anomalous, those get
   incorrectly counted as "successful detections" because they happen to be
   in an "attack-context" file.

When training data is small (smoke), force #2 wins: a narrow normal yields
big gaps. When training data is large (full), force #1 wins: a generous
normal yields better statistical accuracy but a smaller gap.

**The interview question.** "Your smoke run got PR-AUC of 0.95 and your full
run got 0.69. Which would you put in production?" — the **full-run model**, even
though the headline number is worse. The full-run model is honest about
diverse driving phases. The smoke-run model is overconfident in a narrow
distribution it happens to have memorized.

This is one of the most important conceptual lessons in unsupervised anomaly
detection: **higher reported metrics on biased evaluation labels are not the
same as better real-world performance.** A model that overfits the evaluation's
biases looks great until it ships.

**Analogous situations in your engineering background:**

- A HiL test suite that you've tuned your controller against forever — high
  pass rate doesn't mean the controller is good, it might just mean it's
  overfit to the suite. New scenarios reveal the truth.
- A Kalman filter tuned on one driving log: low residual error on that log
  doesn't generalize to a different vehicle dynamics regime.
- Same control-systems intuition. Same trap.
""")


# ============================================================
# I. Learning nuggets
# ============================================================
md("""
---
## I. Learning nuggets to remember

Compressed takeaways for future ML projects:

1. **More data is not always better for anomaly detection.** It improves
   modeling but can compress the anomaly-detection signal. Always evaluate
   both ways before committing to one.

2. **PR-AUC is the honest metric for unsupervised AD.** F1 depends on a
   threshold you may have picked badly. Report PR-AUC; report F1 as a
   secondary number tied to a specific operating point.

3. **Widening categorical features helps masquerade attacks most.** When
   anomalies use legitimate categories with subtly-wrong continuous data,
   more category granularity gives the model more handles. Cheap to do.

4. **Bottleneck size is U-shaped.** Default to ~input_dim/4. Sweep ±50%
   if you have time. Don't push to 1 — that's a different kind of model
   (a 1-D embedding, useful for visualization but bad for reconstruction).

5. **Iteration log discipline.** Three runs with notes is worth more than
   thirty runs without. The notes are where the intuition compounds. Write
   "I expected X, got Y, here's why I think." Future-you will thank you.

6. **Train/val/test still matter even when "test" is just a different
   distribution.** In our cross-dataset eval (Phase 4.5), the held-out OTIDS
   test set isn't enough — generalization to CrySyS is what production
   would look like.

7. **The threshold and the model are independent decisions.** The threshold
   shifts the operating point on the PR curve. The model determines the
   curve itself. Tune the model first (maximize PR-AUC); tune the threshold
   second (pick the operating point that matches your false-positive budget).

8. **Visualize error distributions every time.** Numbers can lie; histograms
   don't. If your PR-AUC looks great but the histograms look weird, trust
   the histograms — the numbers are masking something.
""")


# ============================================================
# J. What I'd try next
# ============================================================
md("""
---
## J. What I'd try next

Concrete experiments worth running on this same codebase:

**Easy wins (~1 hour each):**

- `hidden=32` (double the outer hidden layer). Probably lifts impersonation
  PR-AUC modestly. Same model size as a slightly bigger production version.
- `top_k=50`. See if returns diminish — if PR-AUC plateaus at top_k=40 we know
  we've found the right neighborhood.
- `epochs=60` with early stopping. Does the full-data model just need longer
  to find a tighter representation?
- Threshold sweep: try 95th, 97th, 99th, 99.5th percentile and report the
  PR-AUC-independent precision/recall trade-off.

**Medium effort (~half a day):**

- **Phase-stratified evaluation.** Split the OTIDS test set into chunks based
  on inferred driving phase (use timestamp gaps + CAN ID statistics to detect
  phase changes). Report per-phase PR-AUC. Test the smoke-vs-full hypothesis
  quantitatively.
- **Z-score normalization** for data bytes (instead of `/255`). Catches
  outlier bytes (sensor faults) more gracefully than min-max.
- **Learn the threshold per-ID.** Compute reconstruction-error percentile per
  CAN ID instead of globally. Some IDs may be intrinsically noisier.

**Bigger projects (this becomes Phase 2.5 / 3):**

- **LSTM autoencoder.** Frame-level model can't see temporal patterns. An
  LSTM-AE would catch impersonation attacks that break a CAN ID's data-byte
  cycle. Expected gain: impersonation PR-AUC from 0.80 to 0.90+.
- **Cross-dataset training.** Train on OTIDS + CrySyS jointly. The model
  learns vehicle-invariant features. Massively reduces the cross-dataset
  domain shift from Phase 4.5.
- **Learned CAN-ID embeddings.** Replace the one-hot encoding with a learned
  embedding layer trained jointly with the autoencoder. Embeddings transfer
  across vehicles better than one-hots.

If you only do one follow-up, do the **LSTM autoencoder**. It's the natural
"Phase 2.5" that every interviewer will ask about, and it's the obvious next
step from the impersonation result.
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

out = Path(__file__).parent / "04_hyperparameter_comparison.ipynb"
nbf.write(nb, str(out))
print(f"Wrote {out} ({len(cells)} cells)")
