import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = "/home/groups/comp3710/ADNI"
DEFAULT_SPLIT_FILE = PROJECT_DIR / "splits" / "val_split.json"
DEFAULT_STATS_FILE = PROJECT_DIR / "splits" / "norm_stats.json"

LABELS = {"NC": 0, "AD": 1}  # AD is the positive class
IMG_SIZE = 224
IMG_EXTS = {".jpeg", ".jpg", ".png"}


def parse_name(path: Path):
    pid, slice_no = path.stem.rsplit("_", 1)
    return pid, int(slice_no)


def _scan_folder(folder: Path, label: int, keep_ids=None):
    """List one class folder (AD or NC) as records, optionally keeping only some patients."""
    records = []
    for p in folder.iterdir():
        if p.suffix.lower() not in IMG_EXTS:
            continue
        pid, slice_no = parse_name(p)
        if keep_ids is not None and pid not in keep_ids:
            continue
        records.append({"path": p, "label": label, "pid": pid, "slice_no": slice_no})
    return records


def _add_slice_positions(records):
    """Give each slice its position within its patient, sorting slice numbers numerically."""
    by_pid = defaultdict(list)
    for r in records:
        by_pid[r["pid"]].append(r)
    for patient_records in by_pid.values():
        patient_records.sort(key=lambda r: r["slice_no"])
        for pos, r in enumerate(patient_records):
            r["slice_pos"] = pos
    records.sort(key=lambda r: (r["pid"], r["slice_pos"]))


def check_no_leakage(records_by_split):
    """Assert no patient is in two splits, and every patient has a single label."""
    ids = {name: {r["pid"] for r in recs} for name, recs in records_by_split.items()}
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = ids[a] & ids[b]
            assert not shared, f"Patients in both {a} and {b}: {sorted(shared)[:5]}"
    for name, recs in records_by_split.items():
        labels = defaultdict(set)
        for r in recs:
            labels[r["pid"]].add(r["label"])
        mixed = [pid for pid, ls in labels.items() if len(ls) > 1]
        assert not mixed, f"Patients with both labels in {name}: {mixed[:5]}"


def build_records(root=DEFAULT_ROOT, split_file=DEFAULT_SPLIT_FILE):
    """Return {'train': [...], 'val': [...], 'test': [...]} slice records.

    Train/val membership comes from the saved patient lists; nothing is
    re-randomised here, so every run uses identical splits.
    """
    base = Path(root) / "AD_NC"
    with open(split_file) as f:
        split = json.load(f)

    out = {"train": [], "val": [], "test": []}
    for cls, label in LABELS.items():
        for part in ("train", "val"):
            wanted = set(split[part][cls])
            found = _scan_folder(base / "train" / cls, label, keep_ids=wanted)
            missing = wanted - {r["pid"] for r in found}
            assert not missing, (
                f"{len(missing)} {part}/{cls} patients from split file not found"
            )
            out[part] += found
        out["test"] += _scan_folder(base / "test" / cls, label)

    for recs in out.values():
        _add_slice_positions(recs)
    check_no_leakage(out)
    return out
