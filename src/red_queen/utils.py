"""
Utility functions: config loading, seeding, logging, text sanitization.
"""

import json
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# ── Config ──────────────────────────────────────────────────


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML config and return as nested dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── Seeding ─────────────────────────────────────────────────


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Device ──────────────────────────────────────────────────


def get_device() -> torch.device:
    """Return cuda device if available, else cpu."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("[INFO] Using CPU")
    return dev


# ── Run directory management ────────────────────────────────


def create_run_dir(output_dir: str) -> Path:
    """Create an auto-numbered run directory: outputs/run_001, run_002, ..."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    next_num = len(existing) + 1
    run_dir = output_dir / f"run_{next_num:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save timestamp
    (run_dir / "timestamp.txt").write_text(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[INFO] Run directory: {run_dir}")
    return run_dir


# ── Text sanitization ──────────────────────────────────────

# Common slurs / hate terms to mask in logs (extend as needed)
_MASK_PATTERNS = [
    r"\bn[i1!|]gg[e3a@][r5sz]?\b",
    r"\bf[a@4]gg?[o0][t7]\b",
    r"\bk[i1!]ke\b",
    r"\bch[i1!]nk\b",
    r"\bsp[i1!]c\b",
    r"\btr[a@4]nn[yi1!e3]\b",
    r"\bwh[o0]re\b",
    r"\bb[i1!]tch\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _MASK_PATTERNS]


def sanitize_text(text: str, mask_char: str = "█") -> str:
    """Replace known slurs with mask characters for safe logging."""
    for pattern in _COMPILED:
        text = pattern.sub(lambda m: mask_char * len(m.group()), text)
    return text


def safe_log_example(
    text: str, label: int, action: str = "", sanitize: bool = True, mask_char: str = "█"
) -> str:
    """Format a single example for safe logging."""
    display = sanitize_text(text, mask_char) if sanitize else text
    # Truncate long texts
    if len(display) > 120:
        display = display[:117] + "..."
    label_str = "HATE" if label == 1 else "SAFE"
    parts = [f"[{label_str}]"]
    if action:
        parts.append(f"({action})")
    parts.append(display)
    return " ".join(parts)


# ── Metrics I/O ─────────────────────────────────────────────


def save_metrics(metrics: dict, path: Path):
    """Save metrics dict as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"[INFO] Metrics saved to {path}")


def save_config_snapshot(cfg: dict, run_dir: Path):
    """Save a copy of the config used for this run."""
    path = run_dir / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
