"""OTIDS log file loader.

The OTIDS line format is whitespace-separated with inline label words:

    Timestamp: <ts>  ID: <hex>  000  DLC: <dlc>  <data bytes...>

After `str.split()` the token positions are fixed:

    [0] "Timestamp:"  [1] <ts>     [2] "ID:"   [3] <hex>     [4] "000"
    [5] "DLC:"         [6] <dlc>                [7..] <data bytes>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_otids_file(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Parse one OTIDS log file into a DataFrame.

    Args:
        path: Path to the .txt log file.
        limit: If set, stop after parsing this many lines.

    Returns:
        DataFrame with columns: timestamp, can_id, dlc, data_0..data_7.
        Unused byte slots (when dlc < 8) are NaN.
    """
    rows: list[tuple] = []
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            tokens = line.split()
            if len(tokens) < 7 or tokens[0] != "Timestamp:":
                continue
            ts = float(tokens[1])
            cid = int(tokens[3], 16)
            dlc = int(tokens[6])
            data_bytes = [int(t, 16) for t in tokens[7:7 + dlc]]
            data_padded = data_bytes + [np.nan] * (8 - len(data_bytes))
            rows.append((ts, cid, dlc, *data_padded))

    cols = ["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)]
    return pd.DataFrame(rows, columns=cols)


def find_file(raw_dir: Path, keyword: str) -> Path | None:
    """Return the first file in raw_dir whose name contains `keyword` (case-insensitive)."""
    if not raw_dir.exists():
        return None
    matches = [
        p for p in raw_dir.iterdir()
        if p.is_file() and keyword.lower() in p.name.lower()
    ]
    return matches[0] if matches else None
