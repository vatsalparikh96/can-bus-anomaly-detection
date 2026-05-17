"""CrySyS log file loader.

CrySyS uses the SocketCAN/candump format. Each line:

    (<timestamp>) <interface> <can_id>#<data_bytes_hex>

For example:

    (0.000000) can0 110#02202e1300181300

Parsing notes:
- The timestamp is in parentheses, relative seconds.
- The interface (e.g., "can0") is ignored.
- The CAN ID is hexadecimal, no `0x` prefix.
- Data bytes form a single continuous hex string; DLC is implicit (`len/2`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_crysys_file(path: Path, limit: int | None = None) -> pd.DataFrame:
    """Parse one CrySyS candump log into a DataFrame.

    Returns a DataFrame with the same schema as `load_otids_file`:
    columns timestamp, can_id, dlc, data_0..data_7. Unused byte slots are NaN.
    """
    rows: list[tuple] = []
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line or not line.startswith("("):
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue
            try:
                ts = float(tokens[0].strip("()"))
                msg = tokens[2]
                if "#" not in msg:
                    continue
                cid_str, data_str = msg.split("#", 1)
                cid = int(cid_str, 16)
                dlc = len(data_str) // 2
                if dlc > 8:
                    dlc = 8
                data_bytes = [int(data_str[2 * j:2 * j + 2], 16) for j in range(dlc)]
                data_padded = data_bytes + [np.nan] * (8 - len(data_bytes))
            except (ValueError, IndexError):
                continue
            rows.append((ts, cid, dlc, *data_padded))

    cols = ["timestamp", "can_id", "dlc"] + [f"data_{i}" for i in range(8)]
    return pd.DataFrame(rows, columns=cols)


def find_crysys_benign_logs(logs_dir: Path) -> list[Path]:
    """Return all `*-benign.log` files (the original recorded traces)."""
    return sorted(logs_dir.glob("*/*-benign.log"))


def find_crysys_attack_logs(logs_dir: Path, attack_keyword: str | None = None) -> list[Path]:
    """Return malicious log files, optionally filtered by keyword (e.g. 'ADD-INCR')."""
    pattern = "*/*-malicious-*.log"
    all_attack = sorted(logs_dir.glob(pattern))
    # Exclude the *-inj-messages.log files — those are just the injected frames, not full logs.
    full_logs = [p for p in all_attack if "-inj-messages" not in p.name]
    if attack_keyword:
        full_logs = [p for p in full_logs if attack_keyword.lower() in p.name.lower()]
    return full_logs
