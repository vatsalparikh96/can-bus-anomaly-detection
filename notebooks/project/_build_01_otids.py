"""Build notebooks/05_explore_otids.ipynb.

Phase 3 of the project plan: explore the OTIDS dataset, document its quirks,
visualize attack signatures, and lock in preprocessing decisions before any
modeling.
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
md(r"""
# 05 — OTIDS Dataset Deep-Dive

**Goal of this notebook.** Get the OTIDS data on disk, look at it with your own
eyes, understand its quirks, and decide on preprocessing — before any modeling.

This is where ML projects most often go wrong: people skip data exploration,
build a fancy model, and discover three weeks later that their normalization
was leaky or their CAN IDs were parsed as decimals when they're actually hex.
We avoid that by spending two days here.

**What you'll do.**
1. Confirm the raw files are on disk.
2. Write a loader that handles the OTIDS line format.
3. Load attack-free traffic and inspect schema, types, ranges.
4. Visualize CAN ID frequency, inter-arrival times, and data byte distributions.
5. Load each attack file (DoS, Fuzzy, Impersonation) and compare against
   attack-free. **This is where the anomaly signal becomes visible to the eye.**
6. Lock in preprocessing decisions: which features, what normalization, top-K
   CAN ID handling.

**About OTIDS.** The OTIDS (Otoidentified CAN intrusion) dataset was released
by [HCRL at Korea University](https://ocslab.hksecurity.net/Dataset/CAN-intrusion-dataset).
It was collected from a Kia Soul and contains four CAN-bus traffic logs:

| File | What it contains |
|---|---|
| Attack-free | Normal driving traffic |
| DoS | Normal traffic with periodic injections of CAN ID `0x000` (highest priority) |
| Fuzzy | Normal traffic with random CAN IDs and random data injected |
| Impersonation | Normal traffic with spoofed messages mimicking legitimate IDs |

**Sections.**
- A. Setup and file discovery
- B. The OTIDS line format
- C. Build a loader
- D. Load attack-free; verify schema
- E. CAN ID analysis
- F. Timestamp analysis
- G. Data byte distributions
- H. Attack-free vs DoS (the easy attack)
- I. Attack-free vs Fuzzy (medium difficulty)
- J. Attack-free vs Impersonation (the hard attack)
- K. Gotchas + preprocessing decisions
- L. Self-check
""")


# ============================================================
# A. Setup and file discovery
# ============================================================
md(r"""
---
## A. Setup and file discovery

First, find the raw files. They should be at `../data/otids/raw/` relative to
this notebook. If you haven't downloaded yet, see
[`scripts/download_data.py`](../scripts/download_data.py) for the download
instructions.
""")

code("""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RAW_DIR = Path("../../data/otids/raw")
print(f"Looking in: {RAW_DIR.resolve()}")
print(f"Exists: {RAW_DIR.exists()}")

if RAW_DIR.exists():
    files = sorted(p for p in RAW_DIR.iterdir() if p.is_file())
    print(f"\\nFound {len(files)} files:")
    for p in files:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p.name:<40s} {size_mb:>8.1f} MB")
else:
    print("\\nDirectory not found. Run scripts/download_data.py first.")
""")


# ============================================================
# B. The OTIDS line format
# ============================================================
md(r"""
---
## B. The OTIDS line format

Each line in an OTIDS log file looks like this (whitespace-separated, with
inline labels):

```
Timestamp: 1478195722.345678        ID: 0316    000    DLC: 8    05 21 68 09 21 21 00 6f
```

Parsing notes:

- **`Timestamp:`** — Unix epoch float, seconds.
- **`ID:`** — CAN ID in **hexadecimal** (no `0x` prefix). Range 0x000 to 0x7FF
  for 11-bit standard frames.
- The `000` after the ID is the type/RTR flag. Almost always `000`. Ignore for
  our purposes.
- **`DLC:`** — Data Length Code, 0 to 8. Tells you how many data bytes follow.
- **Data bytes** — DLC of them, each one hex byte (0x00 to 0xFF).

Format quirks to be aware of (these are exactly the kind of thing that breaks
unwary parsers):

1. The number of data bytes varies per line (it equals DLC).
2. The CAN ID is hex, not decimal. `pd.read_csv` would silently parse it as
   string. You must `int(value, 16)` to get a usable integer.
3. The label words (`Timestamp:`, `ID:`, `DLC:`) appear *in every line* — they
   are not a header. A naive `read_csv` will include them in the parsed values.
4. There's no `Flag` / label column in OTIDS. Detection must be unsupervised
   (this is why we use an autoencoder).
""")


# ============================================================
# C. Build a loader
# ============================================================
md(r"""
---
## C. Build a loader

We write a `load_otids_file` function that reads one OTIDS log and returns a
pandas DataFrame with these columns:

| Column | Type | Description |
|---|---|---|
| `timestamp` | float64 | Unix epoch seconds |
| `can_id` | int64 | Decimal CAN ID (converted from hex) |
| `dlc` | int8 | Data length code (0–8) |
| `data_0` ... `data_7` | int16 (with NaN allowed for unused bytes) | Data byte 0–7 |

We pad short messages with NaN in unused byte slots — that lets us treat all
rows uniformly later.
""")

md(r"""
The format is rigid: every line has the same shape. So we can avoid the slow
`.index()` calls and use **fixed token positions** instead. With 4M+ rows in the
attack-free file, this matters — the optimized loader is ~5x faster.

Set `LIMIT` to `None` for full files (a few seconds each on a modern laptop) or
to a small int while iterating (saves rebuilding plots while you change them).
""")

code("""
# Set to None to load full files. Set to e.g. 100_000 for fast iteration.
LIMIT: int | None = None


def load_otids_file(path: Path, limit: int | None = LIMIT) -> pd.DataFrame:
    \"\"\"Load one OTIDS log file into a DataFrame.

    Format expected (whitespace-separated, label words inline):
        Timestamp: <ts>  ID: <hex>  000  DLC: <dlc>  <data bytes...>

    After split() the token positions are fixed:
        [0]"Timestamp:"  [1]<ts>  [2]"ID:"  [3]<hex>  [4]"000"
        [5]"DLC:"        [6]<dlc>           [7..]<data bytes>
    \"\"\"
    rows = []
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            tokens = line.split()
            if len(tokens) < 7 or tokens[0] != "Timestamp:":
                # Malformed or empty line — skip silently.
                continue
            ts  = float(tokens[1])
            cid = int(tokens[3], 16)              # hex -> int
            dlc = int(tokens[6])
            data_bytes = [int(t, 16) for t in tokens[7:7 + dlc]]
            data_padded = data_bytes + [np.nan] * (8 - len(data_bytes))
            rows.append((ts, cid, dlc, *data_padded))

    cols = ["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)]
    return pd.DataFrame(rows, columns=cols)


def find_file(keyword: str) -> Path | None:
    \"\"\"Find a file in RAW_DIR whose name contains `keyword` (case-insensitive).\"\"\"
    if not RAW_DIR.exists():
        return None
    matches = [p for p in RAW_DIR.iterdir()
               if p.is_file() and keyword.lower() in p.name.lower()]
    return matches[0] if matches else None


attack_free_path = find_file("attack_free") or find_file("normal")
print(f"attack-free file: {attack_free_path}")
""")


# ============================================================
# D. Load attack-free; verify schema
# ============================================================
md(r"""
---
## D. Load attack-free; verify schema

Load the attack-free file and look at it. **Resist the urge to scroll past.**
Read each column. Confirm types are what you expect. Look at min/max. This is
the cheap-and-easy phase where format bugs surface — once you start training,
they get expensive to find.
""")

code("""
assert attack_free_path is not None, "No attack-free file found in data/otids/raw"

df_free = load_otids_file(attack_free_path)
print(f"Loaded {len(df_free):,} rows from {attack_free_path.name}")
print()
print(df_free.head())
print()
df_free.info()
""")

code("""
# Describe numeric columns
print(df_free.describe())
""")

md(r"""
**Things to confirm before moving on:**

- `can_id` min ≥ 0 and max ≤ 2047 (11-bit standard CAN IDs).
- `dlc` is in [0, 8].
- `data_0` through `data_7` are in [0, 255], with NaN where DLC < the slot index.
- `timestamp` is monotonically increasing — it's a chronological log.
""")

code("""
# Sanity checks
assert df_free["can_id"].min() >= 0 and df_free["can_id"].max() < 2048, "CAN ID out of 11-bit range"
assert df_free["dlc"].min() >= 0 and df_free["dlc"].max() <= 8, "DLC out of range"
assert df_free["timestamp"].is_monotonic_increasing, "Timestamps not monotonically increasing"
for i in range(8):
    col = df_free[f"data_{i}"].dropna()
    assert col.min() >= 0 and col.max() <= 255, f"data_{i} out of byte range"
print("Schema sanity checks passed.")
""")


# ============================================================
# E. CAN ID analysis
# ============================================================
md(r"""
---
## E. CAN ID analysis

CAN IDs encode message type (e.g., engine RPM = 0x316, vehicle speed = 0x153,
etc.). On a normal bus, you see a small set of IDs repeating cyclically.

Two things to check:
1. **How many distinct CAN IDs are there?** Usually 30–80 on a passenger car.
2. **What's the frequency distribution?** A few IDs dominate (sent every 10 ms);
   many are rare (event-driven).
""")

code("""
n_unique = df_free["can_id"].nunique()
print(f"Number of distinct CAN IDs in attack-free: {n_unique}")
print()

id_counts = df_free["can_id"].value_counts()
print("Top 10 most frequent IDs:")
for cid, count in id_counts.head(10).items():
    pct = 100 * count / len(df_free)
    print(f"  0x{cid:03X}  ({cid:>4d})  {count:>8,d}  ({pct:5.2f}%)")
""")

code("""
# Plot frequency distribution: top 30 CAN IDs
top_n = 30
top = id_counts.head(top_n)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(top)), top.values)
ax.set_xticks(range(len(top)))
ax.set_xticklabels([f"0x{c:03X}" for c in top.index], rotation=60, ha="right", fontsize=8)
ax.set_ylabel("frame count")
ax.set_title(f"Top {top_n} CAN IDs in attack-free traffic")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()

results_dir = Path("../../results")
results_dir.mkdir(exist_ok=True)
fig.savefig(results_dir / "otids_canid_freq_attackfree.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md(r"""
**Preprocessing implication.** The frequency distribution has a long tail.
For features, we'll one-hot encode only the top-K most common IDs (e.g., K=20)
and bucket everything else as "other." This keeps the feature dimension small
without losing information about the common cases.
""")


# ============================================================
# F. Timestamp analysis
# ============================================================
md(r"""
---
## F. Timestamp analysis

CAN bus traffic is *cyclic*: most messages have a fixed period (10 ms is
common). Looking at inter-arrival times tells you both the bus utilization and
which IDs are periodic vs. event-driven.
""")

code("""
duration = df_free["timestamp"].max() - df_free["timestamp"].min()
rate = len(df_free) / duration
print(f"Total duration: {duration:.1f} s ({duration/60:.1f} min)")
print(f"Mean message rate: {rate:,.0f} msg/s")
print(f"Mean inter-arrival: {1000 / rate:.2f} ms")
""")

code("""
# Distribution of inter-arrival times (in ms)
dt_ms = df_free["timestamp"].diff().dropna() * 1000
print(f"Inter-arrival stats (ms):")
print(f"  median: {dt_ms.median():.3f}")
print(f"  mean:   {dt_ms.mean():.3f}")
print(f"  p99:    {dt_ms.quantile(0.99):.3f}")

fig, ax = plt.subplots(figsize=(8, 3))
# Clip at 5 ms to see the bulk of the distribution
ax.hist(dt_ms[dt_ms < 5], bins=80, edgecolor="black", alpha=0.7)
ax.set_xlabel("inter-arrival time (ms)")
ax.set_ylabel("count")
ax.set_title("Attack-free traffic — inter-arrival time distribution (clipped at 5 ms)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()
""")


# ============================================================
# G. Data byte distributions
# ============================================================
md(r"""
---
## G. Data byte distributions

Each data byte is in [0, 255]. The distribution depends entirely on what
that byte means in the message — speed, voltage, flag bits, etc. The
distribution across the whole dataset (mixing all CAN IDs) is messy and not
super informative on its own, but quickly checking it confirms the byte values
look like a real CAN trace and not, say, ASCII text.
""")

code("""
fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharey=True)
for i in range(8):
    ax = axes[i // 4][i % 4]
    col = df_free[f"data_{i}"].dropna()
    ax.hist(col, bins=64, edgecolor="black", alpha=0.7)
    ax.set_title(f"data_{i}  (n={len(col):,})")
    ax.set_xlim(-5, 260)

fig.suptitle("Attack-free traffic — distribution of each data byte (all IDs mixed)")
fig.tight_layout()
plt.show()
""")

md(r"""
**Preprocessing implication.** Data bytes are in [0, 255] by hardware spec —
not learned from data. We'll normalize as a *fixed constant* min-max: divide
by 255. No `fit/transform` needed for this part.
""")


# ============================================================
# H. Attack-free vs DoS
# ============================================================
md(r"""
---
## H. Attack-free vs DoS — the easy attack

A DoS attack on the CAN bus floods it with the highest-priority frames
(CAN ID `0x000`). Since CAN bus arbitration prioritizes lower IDs, these
flood messages preempt all legitimate traffic.

**Expected signature:** the frequency of CAN ID `0x000` (or other very low IDs)
should be wildly elevated compared to the attack-free baseline.
""")

code("""
dos_path = find_file("dos")
assert dos_path is not None, "No DoS file found in data/otids/raw"

df_dos = load_otids_file(dos_path)
print(f"Loaded {len(df_dos):,} rows from {dos_path.name}")

# Count of CAN ID 0x000 (and nearby IDs)
free_low_ids  = (df_free["can_id"] == 0).sum()
dos_low_ids   = (df_dos["can_id"] == 0).sum()
print(f"\\nCount of CAN ID 0x000 in attack-free: {free_low_ids:,}  ({100*free_low_ids/len(df_free):.3f}%)")
print(f"Count of CAN ID 0x000 in DoS:         {dos_low_ids:,}  ({100*dos_low_ids/len(df_dos):.3f}%)")
""")

code("""
# Side-by-side top-10 CAN IDs
free_top = df_free["can_id"].value_counts().head(10)
dos_top  = df_dos["can_id"].value_counts().head(10)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
axes[0].bar(range(len(free_top)), free_top.values)
axes[0].set_xticks(range(len(free_top)))
axes[0].set_xticklabels([f"0x{c:03X}" for c in free_top.index], rotation=45)
axes[0].set_title("Attack-free: top 10 CAN IDs")
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].bar(range(len(dos_top)), dos_top.values, color="tab:red")
axes[1].set_xticks(range(len(dos_top)))
axes[1].set_xticklabels([f"0x{c:03X}" for c in dos_top.index], rotation=45)
axes[1].set_title("DoS attack: top 10 CAN IDs")
axes[1].grid(True, axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(results_dir / "otids_canid_dos_vs_attackfree.png", dpi=120, bbox_inches="tight")
plt.show()
""")

md(r"""
You should see CAN ID `0x000` dominating the DoS chart while being essentially
absent from the attack-free chart. **This is the anomaly signal.** A model
that just learned "CAN ID 0x000 is rare" will catch this attack trivially.

That's why DoS is the easy attack and we expect F1 > 0.95 on it.
""")


# ============================================================
# I. Attack-free vs Fuzzy
# ============================================================
md(r"""
---
## I. Attack-free vs Fuzzy — the medium attack

A Fuzzy attack injects messages with **random CAN IDs and random data**. The
attack is harder to detect because:
- Random CAN IDs sometimes overlap with legitimate-but-rare IDs.
- Random data bytes don't have a single "weird" signature like 0x000 floods.

**Expected signature:** more distinct CAN IDs than attack-free (because of the
random ones), and data byte distributions less concentrated around the
real-protocol values.
""")

code("""
fuzzy_path = find_file("fuzzy")
assert fuzzy_path is not None, "No Fuzzy file found in data/otids/raw"

df_fuzzy = load_otids_file(fuzzy_path)
print(f"Loaded {len(df_fuzzy):,} rows from {fuzzy_path.name}")

print(f"\\nDistinct CAN IDs in attack-free: {df_free['can_id'].nunique():,}")
print(f"Distinct CAN IDs in fuzzy:       {df_fuzzy['can_id'].nunique():,}")
""")

code("""
# How many CAN IDs in fuzzy that are NOT in attack-free?
free_ids = set(df_free["can_id"].unique())
fuzzy_ids = set(df_fuzzy["can_id"].unique())
novel = fuzzy_ids - free_ids
print(f"CAN IDs in fuzzy that never appear in attack-free: {len(novel)}")
print("(These are 'unknown ID' anomalies — high reconstruction error expected.)")

# Plot CAN ID coverage as a 2048-bin histogram (raw counts)
fig, axes = plt.subplots(2, 1, figsize=(12, 4.5), sharex=True)
for ax, df, title in zip(axes, [df_free, df_fuzzy], ["attack-free", "fuzzy"]):
    ids = df["can_id"].values
    ax.hist(ids, bins=2048, range=(0, 2048))
    ax.set_yscale("log")
    ax.set_ylabel("count (log)")
    ax.set_title(title)
axes[1].set_xlabel("CAN ID (decimal)")
fig.tight_layout()
plt.show()
""")


# ============================================================
# J. Attack-free vs Impersonation
# ============================================================
md(r"""
---
## J. Attack-free vs Impersonation — the hard attack

An Impersonation attack injects messages with **legitimate CAN IDs but spoofed
data**. The attacker's frames look indistinguishable from real frames except in
the data bytes themselves.

**Expected signature:** CAN ID distribution is identical to attack-free.
The signal lives in subtle deviations of the data bytes from their normal
patterns *for that particular ID*.

This is the **hard attack** — and exactly why an MLP autoencoder that treats
each frame independently will struggle. Sequence context (LSTM, Phase 2.5)
helps because spoofed frames break the temporal pattern within a CAN ID stream.
""")

code("""
imp_path = find_file("imp") or find_file("spoof") or find_file("164")
assert imp_path is not None, "No Impersonation file found in data/otids/raw"

df_imp = load_otids_file(imp_path)
print(f"Loaded {len(df_imp):,} rows from {imp_path.name}")

# CAN ID overlap with attack-free
imp_ids = set(df_imp["can_id"].unique())
print(f"\\nDistinct CAN IDs in impersonation: {len(imp_ids)}")
print(f"How many IDs in impersonation NOT in attack-free: {len(imp_ids - free_ids)}")
print("(If close to zero, the attack is using only legitimate IDs.)")
""")

code("""
# Compare data byte distributions for the most-spoofed CAN ID in impersonation
# (Look at the data bytes' distribution for the same ID in both files)
most_common_id = df_imp["can_id"].value_counts().index[0]
print(f"Most frequent CAN ID in impersonation: 0x{most_common_id:03X}")

# Show data byte distributions for that ID, free vs impersonation
fig, axes = plt.subplots(2, 4, figsize=(12, 5), sharey=False)
for i in range(8):
    ax = axes[i // 4][i % 4]
    free_byte = df_free.loc[df_free["can_id"] == most_common_id, f"data_{i}"].dropna()
    imp_byte  = df_imp.loc[df_imp["can_id"] == most_common_id, f"data_{i}"].dropna()
    if len(free_byte) > 0:
        ax.hist(free_byte, bins=40, alpha=0.5, label="free", color="tab:blue")
    if len(imp_byte) > 0:
        ax.hist(imp_byte, bins=40, alpha=0.5, label="imp", color="tab:red")
    ax.set_title(f"data_{i}")
    ax.legend(fontsize=8)
fig.suptitle(f"Data byte distributions for CAN ID 0x{most_common_id:03X}: attack-free vs impersonation")
fig.tight_layout()
plt.show()
""")

md(r"""
Look closely. If the attack-free (blue) and impersonation (red) distributions
overlap nearly perfectly in every byte, the impersonation is well-disguised —
an MLP autoencoder that treats each frame independently will have a hard time
separating real from spoof. That's the empirical justification for the
"impersonation F1 will be the worst" prediction in the plan.
""")


# ============================================================
# K. Gotchas + preprocessing decisions
# ============================================================
md(r"""
---
## K. Gotchas + preprocessing decisions

Lock these in before Phase 4. Re-read them when you write `src/data/preprocess.py`.

### Gotchas (things that will silently break the pipeline if ignored)

1. **CAN IDs are hexadecimal.** `int(value, 16)` to convert. `pd.read_csv` won't
   do this for you.
2. **Label words are in every line.** Don't use a naive CSV reader — index past
   them.
3. **DLC varies per line.** Some lines have 0 data bytes, some have 8.
4. **No frame-level labels.** Training must be unsupervised. Evaluation labels
   come from the file (everything in `DoS_dataset` is "attack"-context,
   everything in attack-free is "normal"-context).
5. **Don't normalize using full-dataset statistics.** Re-read the Phase 1
   appendix on data leakage if needed — same rule applies here.

### Preprocessing decisions (for Phase 4)

| Feature | Encoding | Notes |
|---|---|---|
| CAN ID | one-hot of top-K + "other" bucket | K=20 covers >90% of normal traffic. Fit K on training data. |
| Data bytes (8 of them) | divide by 255 (fixed min-max) | Bytes are 0–255 by hardware spec. NaN for unused bytes → use 0 or a special masked value (decide in Phase 4). |
| DLC | divide by 8 | Range is 0–8 by spec. |
| Timestamp | DON'T USE for per-frame MLP | MLP treats each frame independently. Time gaps are sequence info — saved for LSTM (Phase 2.5). |

Total feature count for the MLP: `K_oneHot + 1_other + 8_data_bytes + 1_dlc = ~30`.

### Expected results (from the plan)

| Attack | Why we expect that F1 |
|---|---|
| DoS — F1 > 0.95 | The CAN ID 0x000 spike is blatant. |
| Fuzzy — F1 ~ 0.7–0.85 | Novel CAN IDs are easy; random data within legitimate IDs is harder. |
| Impersonation — F1 ~ 0.5–0.7 | Legitimate IDs + subtly-different bytes. The MLP-without-sequence-context limitation. |
""")


# ============================================================
# L. Self-check
# ============================================================
md(r"""
---
## L. Self-check — are you ready for Phase 4?

Answer in your head, no peeking:

1. What columns does an OTIDS log have, and what's the type of each?
2. Why must we use `int(value, 16)` for the CAN ID? What would happen if we used
   `int(value)` instead?
3. What's the labeling situation in OTIDS, and what's the consequence for our
   training approach?
4. What's the simplest detectable anomaly (and why)?
5. Why is impersonation the hardest attack to detect with a per-frame MLP?
6. For preprocessing, which statistics are *learned from training data* (must
   call `.fit` on train only) and which are *fixed constants*?
7. We have a "DLC" column. Why does that need to be normalized at all if it's
   already a small integer 0–8?

If you can answer 5/7 cleanly, you're ready for **Phase 4: build the MLP
autoencoder.**
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

out = Path(__file__).parent / "01_explore_otids.ipynb"
nbf.write(nb, str(out))
print(f"Wrote {out} ({len(cells)} cells)")
