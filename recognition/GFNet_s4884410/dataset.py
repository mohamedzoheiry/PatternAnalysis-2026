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


def build_transform(mean=None, std=None):
    """Image transform: 224 centre crop, scale to [0, 1], then normalise if stats are given.

    The same deterministic transform is used for every split.
    """
    steps = [T.CenterCrop(IMG_SIZE), T.ToTensor()]  # ToTensor scales to [0, 1]
    if mean is not None:
        steps.append(T.Normalize([mean], [std]))
    return T.Compose(steps)


def compute_norm_stats(train_records, num_workers=4, batch_size=256):
    """Pixel mean and std over training patients only (cropped, no augmentation)."""
    ds = ADNISliceDataset(train_records, build_transform())
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    total, total_sq, n = 0.0, 0.0, 0
    for x, *_ in loader:
        x = x.double()
        total += x.sum().item()
        total_sq += (x**2).sum().item()
        n += x.numel()
    mean = total / n
    std = (total_sq / n - mean**2) ** 0.5
    return mean, std


def load_or_compute_norm_stats(
    train_records, stats_file=DEFAULT_STATS_FILE, num_workers=4
):
    """Load cached stats if they match the current training set, otherwise recompute and save."""
    key = {
        "n_images": len(train_records),
        "n_patients": len({r["pid"] for r in train_records}),
        "img_size": IMG_SIZE,
    }
    stats_file = Path(stats_file)
    if stats_file.exists():
        cached = json.loads(stats_file.read_text())
        if all(cached.get(k) == v for k, v in key.items()):
            return cached["mean"], cached["std"]

    mean, std = compute_norm_stats(train_records, num_workers=num_workers)
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    stats_file.write_text(json.dumps({**key, "mean": mean, "std": std}, indent=2))
    return mean, std


class ADNISliceDataset(Dataset):
    """One item per slice: (image [1, 224, 224], label, patient_id, slice_pos)."""

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def load_image(self, i):
        with Image.open(self.records[i]["path"]) as im:
            img = im.convert("L")
        return self.transform(img)

    def __getitem__(self, i):
        r = self.records[i]
        return self.load_image(i), r["label"], r["pid"], r["slice_pos"]


class ADNIPatientDataset(Dataset):
    """One item per patient: (images [20, 1, 224, 224] in slice order, label, patient_id)."""

    def __init__(self, slice_dataset: ADNISliceDataset):
        self.slices = slice_dataset
        groups = defaultdict(list)
        for i, r in enumerate(slice_dataset.records):
            groups[r["pid"]].append(i)
        recs = slice_dataset.records
        self.pids = sorted(groups)
        self.index = {
            pid: sorted(idx, key=lambda i: recs[i]["slice_pos"])
            for pid, idx in groups.items()
        }

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, k):
        pid = self.pids[k]
        idx = self.index[pid]
        images = torch.stack([self.slices.load_image(i) for i in idx])
        label = self.slices.records[idx[0]]["label"]
        return images, label, pid


def get_datasets(
    root=DEFAULT_ROOT,
    split_file=DEFAULT_SPLIT_FILE,
    stats_file=DEFAULT_STATS_FILE,
    num_workers=4,
):
    """Build train/val/test slice datasets sharing one set of training-only norm stats.

    Returns (datasets, (mean, std)).
    """
    records = build_records(root, split_file)
    mean, std = load_or_compute_norm_stats(records["train"], stats_file, num_workers)
    tf = build_transform(mean, std)
    datasets = {name: ADNISliceDataset(recs, tf) for name, recs in records.items()}
    return datasets, (mean, std)


def get_loaders(datasets, batch_size=64, num_workers=4):
    """Slice-level loaders. Only train is shuffled."""
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def _print_split_sizes(datasets):
    print("\n== Split sizes ==")
    print(f"{'split':6} {'patients':>8} {'AD':>5} {'NC':>5} {'slices':>7}")
    for name, ds in datasets.items():
        label_of = {r["pid"]: r["label"] for r in ds.records}
        ad = sum(label_of.values())
        print(
            f"{name:6} {len(label_of):8d} {ad:5d} {len(label_of) - ad:5d} {len(ds):7d}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--split-file", default=DEFAULT_SPLIT_FILE)
    ap.add_argument("--stats-file", default=DEFAULT_STATS_FILE)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--check-crop", action="store_true", help="check the crop on train/val images"
    )
    ap.add_argument("--grid-out", default=PROJECT_DIR / "images" / "sample_slices.png")
    args = ap.parse_args()

    datasets, (mean, std) = get_datasets(
        args.root, args.split_file, args.stats_file, num_workers=args.num_workers
    )
    _print_split_sizes(datasets)
    print("PASS: no patient-level leakage between train, val and test")
    print(f"\nNormalisation (training patients only): mean={mean:.4f} std={std:.4f}")

    x, y, pid, pos = datasets["train"][0]
    print(
        f"\nSlice sample: image {tuple(x.shape)}, label {y}, patient {pid}, slice pos {pos}"
    )
    images, y, pid = ADNIPatientDataset(datasets["val"])[0]
    print(f"Patient sample: images {tuple(images.shape)}, label {y}, patient {pid}")
