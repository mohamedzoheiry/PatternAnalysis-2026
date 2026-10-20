import argparse
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")  # no display on Rangpur; write to a file instead
import matplotlib.pyplot as plt
from PIL import Image
 
 
def patient_slices(folder: Path, pid: str):
    """Return this patient's slice paths sorted by slice number (numeric, not text)."""
    paths = [p for p in folder.glob(f"{pid}_*") if p.suffix.lower() in {".jpeg", ".jpg", ".png"}]
    return sorted(paths, key=lambda p: int(p.stem.rsplit("_", 1)[1]))
 
 
def first_pid(folder: Path):
    """Smallest patient ID in a class folder."""
    return min(p.stem.rsplit("_", 1)[0] for p in folder.iterdir() if "_" in p.stem)
 
 
def plot_patient(paths, title, out, cols=5):
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, p in zip(axes.flat, paths):
        ax.imshow(Image.open(p).convert("L"), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"slice {p.stem.rsplit('_', 1)[1]}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/groups/comp3710/ADNI")
    ap.add_argument("--split", default="train")
    ap.add_argument("--pid", help="patient ID; requires --cls")
    ap.add_argument("--cls", choices=["AD", "NC"])
    ap.add_argument("--out-dir", default="images")
    args = ap.parse_args()
 
    base = Path(args.root) / "AD_NC" / args.split
    targets = [(args.cls, args.pid)] if args.pid else [(c, first_pid(base / c)) for c in ["AD", "NC"]]
 
    for cls, pid in targets:
        paths = patient_slices(base / cls, pid)
        nums = [int(p.stem.rsplit("_", 1)[1]) for p in paths]
        gaps = [b - a for a, b in zip(nums, nums[1:])]
        print(f"{cls} patient {pid}: {len(paths)} slices, numbers {nums[0]}..{nums[-1]}, "
              f"steps between slices: {sorted(set(gaps))}")
        plot_patient(paths, f"{cls} patient {pid} ({args.split})",
                     Path(args.out_dir) / f"patient_slices_{cls}_{pid}.png")
 
 
if __name__ == "__main__":
    main()
