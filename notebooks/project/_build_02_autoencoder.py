"""Build notebooks/06_autoencoder_first_pass.ipynb.

Phase 4 of the project plan: prototype the MLP autoencoder end-to-end in a
single notebook, on real OTIDS data, before refactoring into `src/` modules.

Following the plan's "implement from scratch first, abstract later" rule —
once the notebook produces real numbers, the refactor into modules is mechanical.
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
# 06 — Autoencoder First Pass (Phase 4)

**Goal of this notebook.** Get a working MLP autoencoder, trained on real
attack-free OTIDS traffic, that detects DoS / Fuzzy / Impersonation attacks
above chance. Everything written inline — no `src/` imports yet. Once the
numbers look right, we refactor.

**Architecture (decided in Phase 3):**

```
Input  (30)  →  Linear(30,16)  →  ReLU
                →  Linear(16, 8)  →  ReLU       (bottleneck)
                →  Linear( 8,16)  →  ReLU
                →  Linear(16,30)  →  Sigmoid    (output in [0,1])
```

- **Input** = 30 features per CAN frame:
  - 20 = one-hot of top-20 most common CAN IDs (learned from training data)
  - 1  = "other" bucket for any CAN ID outside the top-20
  - 8  = normalized data bytes (`data_0 ... data_7`, each `/255`)
  - 1  = normalized DLC (`/8`)
- **Loss** = MSE between input and reconstruction (the autoencoder objective).
- **Optimizer** = Adam, lr = 1e-3.
- **Batch size** = 256.
- **Epochs** = 30 with early stopping (patience = 5 on val loss).

**Sections.**
- A. Setup
- B. Load attack-free + Phase 3 loader
- C. Train / val / test split
- D. `FeatureBuilder` — fit on train, transform any DataFrame
- E. `CANFrameDataset` + `DataLoader`
- F. `MLPAutoencoder` model
- G. Training loop (with early stopping)
- H. Evaluate reconstruction error on attack-free test + attack files
- I. Threshold + metrics (precision / recall / F1 / PR-AUC)
- J. Visualizations
- K. Self-check before refactor
""")


# ============================================================
# A. Setup
# ============================================================
md("""
---
## A. Setup

Imports, device selection, reproducibility seed, and the path constants.
""")

code("""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Reproducibility — set seeds for both numpy and torch
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Paths
RAW_DIR = Path("../../data/otids/raw")
RESULTS_DIR = Path("../../results")
RESULTS_DIR.mkdir(exist_ok=True)

# Cap for fast iteration. Set to None for full files.
# Loading attack-free fully is ~40s; with LIMIT it's a few seconds.
LIMIT: int | None = 500_000
""")


# ============================================================
# B. Phase 3 loader
# ============================================================
md("""
---
## B. Phase 3 loader

Same `load_otids_file` you wrote in Phase 3. Pasted here so this notebook is
self-contained.
""")

code("""
def load_otids_file(path: Path, limit: int | None = LIMIT) -> pd.DataFrame:
    \"\"\"Load one OTIDS log file into a DataFrame.\"\"\"
    rows = []
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            tokens = line.split()
            if len(tokens) < 7 or tokens[0] != "Timestamp:":
                continue
            ts  = float(tokens[1])
            cid = int(tokens[3], 16)
            dlc = int(tokens[6])
            data_bytes = [int(t, 16) for t in tokens[7:7 + dlc]]
            data_padded = data_bytes + [np.nan] * (8 - len(data_bytes))
            rows.append((ts, cid, dlc, *data_padded))
    cols = ["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)]
    return pd.DataFrame(rows, columns=cols)


t0 = time.time()
df_free = load_otids_file(RAW_DIR / "Attack_free_dataset.txt")
print(f"Loaded {len(df_free):,} attack-free rows in {time.time() - t0:.1f}s")
print(df_free.head(3))
""")


# ============================================================
# C. Train / val / test split
# ============================================================
md("""
---
## C. Train / val / test split

### C.1 — What train, val, test mean

You split your data into three disjoint subsets, each with a different job:

| Split | Fraction | The model uses it to... | When |
|---|---|---|---|
| **Train** | 70% | *learn the weights* — gradients computed on this data, parameters update | every step of training |
| **Val** (validation) | 15% | *peek at generalization* — check overfitting, decide when to stop, pick hyperparameters | once per epoch, in a `no_grad()` block |
| **Test** | 15% | *get an honest final number* — never touched during development | exactly once, at the very end |

**Crucial rule.** Only train data updates weights. Val and test are observed
but never used for learning.

**Engineering analogy.** Map this to control-system development:
- **Train** = your test rig. You tune freely: change gains, retry, observe.
- **Val** = a second, separate test rig. You move your tuned controller over
  periodically and check: "Did my tuning generalize, or did I just overfit to
  rig 1's quirks?" Bad val performance while train is good = overfitting.
- **Test** = the customer's car. You only deploy once. If it fails there, you
  don't get to retune — your reputation already took the hit. **You're not
  allowed to look at this data while developing.**

**Why three splits and not two?** With only train + test, you have no early
warning for overfitting. You'd tune-tune-tune, then check test, see disaster,
tune more — and now your "test" set has effectively become val (because you
used it to make development decisions). Reported test numbers stop being
honest. The third split exists to preserve that honesty.

**What this looks like in our training loop (Section G):**

```python
train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer)       # backprop ON
val_loss   = run_one_epoch(model, val_loader,   loss_fn, optimizer=None)  # backprop OFF
```

The asymmetry is the point. Train runs *with* an optimizer (weights update).
Val runs *without* one (compute loss, no learning). That's the train/val
distinction in code.

Then early stopping:

```python
if val_loss < best_val:
    best_state = current_weights
    epochs_no_improve = 0
else:
    epochs_no_improve += 1
if epochs_no_improve >= patience:
    break
```

Val is *not allowed* to influence training directly (no gradients) but *is
allowed* to influence when training stops. Clean division of responsibility.

**Where each split shows up in this notebook:**
- **Train** — Section G (weight updates) and Section I (computing the threshold).
- **Val** — Section G only (early stopping).
- **Test** — Section H (the attack-free baseline against which attack files are compared).
""")

md("""
### C.2 — Why chronological, not random

We split the **attack-free** data 70 / 15 / 15 *chronologically* — by timestamp,
not by random shuffle.

⚠ **Why chronological and not random?** CAN bus data is sequential — adjacent
frames are highly correlated. Random shuffling would leak training-time-step
information into val/test (e.g., a val frame that's 10 ms after a train frame
is nearly identical). For an honest evaluation we slice along the timestamp
axis.

This is a subtler form of the "data leakage" rule from Phase 1's appendix:
**any preprocessing or splitting decision that uses information from the test
set inflates your reported metrics.**
""")

code("""
n = len(df_free)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
# n_test = n - n_train - n_val

# Chronological slicing (rows are already in timestamp order)
df_train = df_free.iloc[:n_train].reset_index(drop=True)
df_val   = df_free.iloc[n_train:n_train + n_val].reset_index(drop=True)
df_test  = df_free.iloc[n_train + n_val:].reset_index(drop=True)

print(f"train: {len(df_train):>9,} rows")
print(f"val:   {len(df_val):>9,} rows")
print(f"test:  {len(df_test):>9,} rows")

# Quick sanity: timestamps must not overlap between splits
print(f"\\ntrain timestamps: [{df_train['timestamp'].min():.3f}, {df_train['timestamp'].max():.3f}]")
print(f"val   timestamps: [{df_val['timestamp'].min():.3f}, {df_val['timestamp'].max():.3f}]")
print(f"test  timestamps: [{df_test['timestamp'].min():.3f}, {df_test['timestamp'].max():.3f}]")
""")


# ============================================================
# D. FeatureBuilder
# ============================================================
md("""
---
## D. `FeatureBuilder` — fit on train, transform anything

This is the Phase 1 min-max exercise generalized. The class has two
responsibilities:

1. **`fit(df)`** — learn parameters *from training data only*:
   - The list of top-K CAN IDs (everything else gets bucketed as "other").
2. **`transform(df)`** — produce a `(n_frames, 30)` float32 NumPy array, using
   the parameters learned at fit time.

Things that are *not* learned (because they're hardware constants):

- Data bytes are 0–255 → divide by 255.
- DLC is 0–8 → divide by 8.

The vectorized one-hot via a 2048-element lookup table is much faster than a
per-row dict lookup — important when transforming 3M+ rows.
""")

code("""
class FeatureBuilder:
    \"\"\"Convert CAN frames into a (n, 30) float32 feature matrix.

    Learned parameters (from fit):
      - top_ids: list of the top_k most common CAN IDs in training data.

    Fixed transforms (no fitting):
      - One-hot encode CAN ID (top-k columns) + 'other' bucket.
      - Data bytes / 255 (NaN -> 0 for unused bytes).
      - DLC / 8.

    Output feature layout for top_k=20:
      cols  0..19  -> one-hot for top-20 CAN IDs
      col   20      -> 'other' bucket
      cols  21..28 -> data_0/255 ... data_7/255
      col   29      -> dlc / 8
    \"\"\"

    def __init__(self, top_k: int = 20):
        self.top_k = top_k
        self.top_ids: list[int] | None = None
        # Lookup table: can_id -> column index. Filled in fit().
        self._lookup: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return self.top_k + 1 + 8 + 1

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        \"\"\"Learn top_k most common CAN IDs from training df.\"\"\"
        counts = df["can_id"].value_counts()
        self.top_ids = counts.head(self.top_k).index.tolist()
        # Lookup table maps every possible 11-bit ID to its column index.
        # Default = top_k (the 'other' bucket).
        self._lookup = np.full(2048, self.top_k, dtype=np.int32)
        for col, cid in enumerate(self.top_ids):
            self._lookup[cid] = col
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        \"\"\"Apply fitted preprocessing. Returns shape (len(df), n_features).\"\"\"
        if self._lookup is None:
            raise RuntimeError("FeatureBuilder not fitted — call .fit() first.")
        n = len(df)
        X = np.zeros((n, self.n_features), dtype=np.float32)

        # One-hot CAN IDs (vectorized)
        can_ids = df["can_id"].to_numpy()
        cols = self._lookup[can_ids]
        X[np.arange(n), cols] = 1.0

        # Data bytes / 255 (fill NaN with 0)
        for j in range(8):
            X[:, self.top_k + 1 + j] = df[f"data_{j}"].fillna(0).to_numpy() / 255.0

        # DLC / 8
        X[:, -1] = df["dlc"].to_numpy() / 8.0

        return X


# Fit on train ONLY
fb = FeatureBuilder(top_k=20)
fb.fit(df_train)
print(f"Top-20 CAN IDs learned from train (decimal):")
print(f"  {fb.top_ids}")
print(f"n_features = {fb.n_features}")

# Transform all three splits
X_train = fb.transform(df_train)
X_val   = fb.transform(df_val)
X_test  = fb.transform(df_test)
print(f"\\nshape  train: {X_train.shape}")
print(f"shape  val:   {X_val.shape}")
print(f"shape  test:  {X_test.shape}")
print(f"dtype:        {X_train.dtype}")
print(f"value range:  [{X_train.min():.3f}, {X_train.max():.3f}]   (should be [0, 1])")
""")

md("""
**Sanity check** on the FeatureBuilder output:

- Each row's first 21 entries (the CAN ID one-hot block) should sum to exactly 1.0 — every frame has *one* CAN ID.
- All values should be in [0, 1] because of the normalization.
- The "other" bucket (column 20) should be 1.0 for CAN IDs that aren't in the top-20, 0 otherwise.
""")

code("""
# Verify one-hot constraint
onehot_block = X_train[:, :fb.top_k + 1]
assert np.allclose(onehot_block.sum(axis=1), 1.0), "Each row's one-hot block must sum to 1.0"

# How many train frames fell into 'other'?
other_frac = X_train[:, fb.top_k].mean()
print(f"Fraction of train frames in 'other' bucket: {other_frac*100:.2f}%")
print("(should be small — top-20 covers most normal traffic)")

assert X_train.min() >= 0 and X_train.max() <= 1, "Values must be in [0, 1]"
print("\\nSanity checks passed.")
""")


# ============================================================
# E. Dataset + DataLoader
# ============================================================
md("""
---
## E. `CANFrameDataset` + `DataLoader`

Same pattern you wrote in Phase 2 (L.4). One sample = one row of the feature
matrix. The Dataset is trivial — most of the work was in the FeatureBuilder.

We wrap each split in a DataLoader with `batch_size=256`. The training loader
gets `shuffle=True`; val and test stay in order so the metrics are reproducible.
""")

code("""
class CANFrameDataset(Dataset):
    \"\"\"PyTorch Dataset wrapping a NumPy feature matrix.\"\"\"

    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.X[idx]


BATCH_SIZE = 256

train_ds = CANFrameDataset(X_train)
val_ds   = CANFrameDataset(X_val)
test_ds  = CANFrameDataset(X_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"train batches: {len(train_loader)}")
print(f"val   batches: {len(val_loader)}")
print(f"test  batches: {len(test_loader)}")
""")


# ============================================================
# F. MLPAutoencoder
# ============================================================
md("""
---
## F. The model

`nn.Sequential` of Linear/ReLU layers. The decoder ends in `nn.Sigmoid` because
all features are in [0, 1] after normalization — the model's output range must
match.

Inspect the printed model. Confirm the layer dimensions are what you expect.
""")

code("""
class MLPAutoencoder(nn.Module):
    \"\"\"MLP autoencoder for CAN frame anomaly detection.

    Architecture:
        input (n_features)
        -> Linear(n_features, 16) -> ReLU
        -> Linear(16, 8) -> ReLU                 # bottleneck
        -> Linear(8, 16) -> ReLU
        -> Linear(16, n_features) -> Sigmoid     # output in [0, 1]
    \"\"\"

    def __init__(self, n_features: int = 30, hidden: int = 16, bottleneck: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


model = MLPAutoencoder(n_features=fb.n_features).to(DEVICE)
print(model)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"\\nTotal parameters: {n_params:,}")
""")


# ============================================================
# G. Training
# ============================================================
md("""
---
## G. Training loop

Standard PyTorch loop from Phase 2 (the 3-line dance), with two extras:

1. **Per-epoch validation pass** — uses `model.eval()` + `torch.no_grad()`.
2. **Early stopping** — if val loss doesn't improve for `patience` epochs, stop.
   Keep the best model weights in memory.

Training time on GPU at LIMIT=500_000: roughly 30–60 seconds total.
""")

code("""
def run_one_epoch(model, loader, loss_fn, optimizer=None) -> float:
    \"\"\"One pass through the loader. If optimizer is provided, trains.\"\"\"
    is_training = optimizer is not None
    model.train() if is_training else model.eval()
    total_loss = 0.0
    total_n = 0

    grad_ctx = torch.enable_grad() if is_training else torch.no_grad()
    with grad_ctx:
        for batch in loader:
            batch = batch.to(DEVICE, non_blocking=True)
            pred = model(batch)
            loss = loss_fn(pred, batch)        # AE: target = input
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * batch.size(0)
            total_n += batch.size(0)

    return total_loss / total_n


def train(model, train_loader, val_loader, n_epochs=30, lr=1e-3, patience=5):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_loss = run_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss   = run_one_epoch(model, val_loader,   loss_fn, optimizer=None)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val - 1e-6
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        flag = " *" if improved else ""
        print(f"epoch {epoch:2d}  train={train_loss:.6f}  val={val_loss:.6f}"
              f"  ({time.time() - t0:.1f}s){flag}")

        if epochs_no_improve >= patience:
            print(f"Early stopping — no val improvement for {patience} epochs.")
            break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\\nRestored best model weights (val={best_val:.6f})")

    return history


# Train
history = train(model, train_loader, val_loader, n_epochs=30, lr=1e-3, patience=5)
""")

code("""
# Plot loss curves
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(history["train_loss"], label="train")
ax.plot(history["val_loss"],   label="val")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE loss")
ax.set_title("Autoencoder training — MSE reconstruction loss")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "phase4_loss_curve.png", dpi=120, bbox_inches="tight")
plt.show()
""")


# ============================================================
# H. Reconstruction error on attack files
# ============================================================
md("""
---
## H. Reconstruction error on attack files

Now we test the trained model on the three attack files. The expectation:

- **Attack-free test split** — the model has *not* seen these specific frames,
  but they come from the same distribution. Reconstruction error should be low.
- **DoS** — CAN ID 0x000 floods. These frames hit the "other" bucket (since
  0x000 isn't in the top-20 normal IDs). Reconstruction error should be high.
- **Fuzzy** — random CAN IDs + random data. Mix of "other" frames and frames
  with weird data bytes. Reconstruction error should be high.
- **Impersonation** — legitimate CAN IDs, subtly different data. Reconstruction
  error should be *moderately* high — but lower than the others. This is the
  hard attack and is what justifies an LSTM (Phase 2.5).
""")

code("""
def compute_recon_errors(model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    \"\"\"Per-frame MSE reconstruction error.\"\"\"
    model.eval()
    errors = np.empty(len(X), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
            pred = model(batch)
            mse = ((pred - batch) ** 2).mean(dim=1)
            errors[i:i + batch_size] = mse.cpu().numpy()
    return errors


# Errors on attack-free test (baseline)
err_test = compute_recon_errors(model, X_test)
print(f"Attack-free test  : n={len(err_test):,}  mean={err_test.mean():.6f}  p99={np.quantile(err_test, 0.99):.6f}")
""")

code("""
# Load each attack file, build features (with the SAME fitted FeatureBuilder!),
# compute reconstruction errors.
attack_files = {
    "dos":           RAW_DIR / "DoS_attack_dataset.txt",
    "fuzzy":         RAW_DIR / "Fuzzy_attack_dataset.txt",
    "impersonation": RAW_DIR / "Impersonation_attack_dataset.txt",
}

attack_data = {}
for name, path in attack_files.items():
    t0 = time.time()
    df = load_otids_file(path)
    X = fb.transform(df)
    errors = compute_recon_errors(model, X)
    attack_data[name] = {"df": df, "X": X, "errors": errors}
    print(f"{name:>14}: n={len(errors):,}  mean={errors.mean():.6f}  "
          f"p99={np.quantile(errors, 0.99):.6f}  ({time.time() - t0:.1f}s)")
""")


# ============================================================
# I. Threshold + metrics
# ============================================================
md("""
---
## I. Threshold + metrics

### I.1 — What precision, recall, F1, PR-AUC mean

These four metrics summarize how well our classifier separates anomalies from
normals. All four are built from a single foundation: the confusion matrix.

**The four outcomes.** Every prediction falls into one of four boxes:

|  | Model says **anomaly** (positive) | Model says **normal** (negative) |
|---|---|---|
| **Is actually anomaly** | TP (true positive) ✓ | FN (false negative) ✗ — missed |
| **Is actually normal** | FP (false positive) ✗ — false alarm | TN (true negative) ✓ |

Every metric below is built from these four counts.

**Concrete example.** Imagine 1000 normal + 100 attack frames. The model flags
50 frames total. Of those 50: 45 are real attacks (TP), 5 are false alarms (FP).
Of the 50 unflagged: 55 attacks were missed (FN), 995 normals correctly passed
(TN). We'll use these numbers throughout.

---

**Precision = TP / (TP + FP)**

"Of everything I flagged, what fraction was actually an anomaly?"

Example: `45 / (45 + 5) = 0.90`. When the model flags something, it's right 90%
of the time.

Precision cares only about your *flagged* frames. It doesn't care what you missed.

*ADAS analogy:* an emergency-braking system's precision is "how often AEB
triggers correctly when it triggers." Low precision = false brakes in normal
traffic, erodes user trust, dangerous in itself.

---

**Recall = TP / (TP + FN)**

"Of all the actual anomalies that exist, what fraction did I catch?"

Example: `45 / (45 + 55) = 0.45`. The model catches 45% of real attacks.

Recall cares only about real anomalies — and whether you caught them. It
doesn't care about false alarms.

*ADAS analogy:* AEB's recall is "how often AEB triggers when there really is an
emergency." Low recall = missed pedestrians. This is what kills people. In ISO
26262 terms, recall maps to ASIL safety goals.

---

**The precision-recall tradeoff.**

Precision and recall fight each other:
- Lower the threshold (flag more) → recall ↑, precision ↓
- Raise the threshold (flag fewer) → precision ↑, recall ↓

You can trivially get 100% precision (flag nothing) or 100% recall (flag
everything). Neither is useful. Real systems need both reasonably high — and
choosing how to trade them is a product decision, not just a model decision.

---

**F1 = 2 × (P × R) / (P + R)** — the harmonic mean

A single number that's only high when *both* precision and recall are high.

Example: `2 × (0.90 × 0.45) / (0.90 + 0.45) = 0.81 / 1.35 = 0.60`.

*Why harmonic, not arithmetic?* With precision = 1.0 and recall = 0.01:
- Arithmetic mean = 0.505 — "looks OK" but the model catches 1% of attacks
- Harmonic mean = `2 × (1.0 × 0.01) / 1.01 = 0.02` — correctly says "terrible"

Harmonic mean punishes the extremes. **Rule of thumb: F1 is only as good as
the worse of precision and recall.**

---

**PR-AUC — area under the precision-recall curve**

F1 reports performance at *one* threshold. PR-AUC asks: across *all* possible
thresholds, how good is the model's ranking?

Computation:
1. Sort all frames by reconstruction error (highest = most anomalous).
2. Sweep a threshold from above-max to below-min.
3. At each threshold, compute (precision, recall) → one point on a curve.
4. The curve plots precision (y) vs. recall (x). Random model = horizontal
   line at `y = positive-fraction`. Perfect model = the corner (1, 1).
5. **PR-AUC = area under that curve.** Range [0, 1]; higher = better.

The key property: **PR-AUC is threshold-independent.** It measures the model's
intrinsic ability to separate positives from negatives, regardless of which
threshold you happened to pick.

That's why **PR-AUC is the headline metric for unsupervised anomaly
detection**: F1 depends on a threshold choice that may be wrong; PR-AUC doesn't.

---

### I.2 — Reading the metrics table you're about to produce

When you see the table from the next code cell, interpret it this way:

- **High precision, low recall** (likely DoS and Impersonation in your run):
  the model is very picky — most of what it flags is real, but it doesn't flag
  enough. Usually a threshold-position issue.
- **High PR-AUC** (likely 0.85+ for all three attacks): the model *can*
  separate anomalies from normals well. F1 may still be low because of
  threshold choice or labeling artifacts.
- **F1 alone is misleading** when class imbalance is severe, labels are coarse
  (our case — file-level not frame-level), or the threshold is in the wrong
  place.

**In our setup, expect:** high precision, high PR-AUC, *low* recall and F1.
That's a labeling problem with OTIDS (per the caveat below) — not a model
problem. **PR-AUC is the honest metric.**

---

### I.3 — Picking a threshold and computing labels

Pick a threshold at the **99th percentile of training reconstruction error**.
Frames with error above the threshold are flagged as anomalies.

For each attack file, we treat:
- attack-free test frames as **negatives** (label 0),
- attack-file frames as **positives** (label 1, "this came from an attack
  scenario").

Then we compute precision / recall / F1 / PR-AUC.

**Caveat**: this labeling is *coarse*. OTIDS attack files contain a mix of
normal traffic and injected attack frames — but we don't have per-frame labels,
so we treat the whole file as "attack-context". Reported F1 is therefore a
lower bound on the true detection performance you'd get with per-frame labels.
PR-AUC is less affected because it's threshold-independent.
""")

code("""
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

# Threshold from training error (use train, not val, to keep val honest)
err_train = compute_recon_errors(model, X_train)
threshold = float(np.quantile(err_train, 0.99))
print(f"Threshold (99th percentile of train errors): {threshold:.6f}")
print(f"On attack-free test: {(err_test > threshold).mean()*100:.2f}% flagged "
      f"(should be ~1%)")
""")

code("""
def evaluate(err_neg: np.ndarray, err_pos: np.ndarray, threshold: float) -> dict:
    errors = np.concatenate([err_neg, err_pos])
    labels = np.concatenate([np.zeros(len(err_neg)), np.ones(len(err_pos))])
    preds = (errors > threshold).astype(int)
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "pr_auc":    average_precision_score(labels, errors),
    }


metrics = {}
print(f"{'attack':<16s} {'precision':>10s} {'recall':>10s} {'F1':>10s} {'PR-AUC':>10s}")
print("-" * 60)
for name, data in attack_data.items():
    m = evaluate(err_test, data["errors"], threshold)
    metrics[name] = m
    print(f"{name:<16s} {m['precision']:>10.3f} {m['recall']:>10.3f} "
          f"{m['f1']:>10.3f} {m['pr_auc']:>10.3f}")

# Save metrics to JSON
with open(RESULTS_DIR / "phase4_metrics.json", "w") as f:
    json.dump({"threshold": threshold, **metrics}, f, indent=2)
print(f"\\nSaved to {RESULTS_DIR / 'phase4_metrics.json'}")
""")


# ============================================================
# J. Visualizations
# ============================================================
md("""
---
## J. Visualizations

Three plots:

1. **Reconstruction error histograms** — attack-free vs. each attack, log scale.
   The amount of separation between the histograms is the visual signature of
   how detectable each attack is.
2. **Precision-recall curves** — one per attack. Far above the y=0 line = good.
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (name, data) in zip(axes, attack_data.items()):
    bins = np.linspace(0, max(err_test.max(), data["errors"].max()) * 1.05, 80)
    ax.hist(err_test,        bins=bins, alpha=0.6, label="attack-free test",  color="tab:blue")
    ax.hist(data["errors"],  bins=bins, alpha=0.6, label=name,                color="tab:red")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1, label="threshold")
    ax.set_yscale("log")
    ax.set_xlabel("reconstruction error")
    ax.set_title(name)
    ax.legend(fontsize=8)
axes[0].set_ylabel("count (log)")
fig.suptitle("Reconstruction error distributions")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "phase4_recon_error_hist.png", dpi=120, bbox_inches="tight")
plt.show()
""")

code("""
from sklearn.metrics import precision_recall_curve

fig, ax = plt.subplots(figsize=(7, 5))
for name, data in attack_data.items():
    errors = np.concatenate([err_test, data["errors"]])
    labels = np.concatenate([np.zeros(len(err_test)), np.ones(len(data["errors"]))])
    precision, recall, _ = precision_recall_curve(labels, errors)
    ax.plot(recall, precision, label=f"{name} (AP={metrics[name]['pr_auc']:.3f})")
ax.set_xlabel("recall")
ax.set_ylabel("precision")
ax.set_title("Precision-recall curves — per attack type")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "phase4_pr_curves.png", dpi=120, bbox_inches="tight")
plt.show()
""")


# ============================================================
# K. Self-check
# ============================================================
md("""
---
## K. Self-check + write down findings

Answer these in your head, then write 2-3 sentences in
`../results/iteration_log.md` (creating the file if it doesn't exist):

1. **DoS F1 should be very high (>0.95).** Is it? If not, why?
2. **Impersonation F1 should be lower** than DoS and Fuzzy. Is it?
3. **Attack-free test false-positive rate** should be ~1% (since threshold is
   the 99th percentile of *train* errors and val/test are from the same
   distribution). Is it?
4. **What surprised you?**

Then — **only if all sanity checks pass** — we're ready to refactor this
notebook into `src/` modules.

If something looks wrong, common culprits:
- LIMIT too small (model didn't see enough variety in training).
- FeatureBuilder fit on full df (data leakage) instead of just train.
- Threshold computed on val/test instead of train.
- Forgot `model.eval()` before evaluation (rare here without Dropout, but build the habit).
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

out = Path(__file__).parent / "02_autoencoder.ipynb"
nbf.write(nb, str(out))
print(f"Wrote {out} ({len(cells)} cells)")
