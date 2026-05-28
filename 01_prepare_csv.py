from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.biodram.utils import load_config, ensure_dir, set_seed


def collect_images(root: Path, class_names):
    rows = []
    for label, name in enumerate(class_names):
        folder = root / name
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                rows.append({"path": str(p.resolve()), "label": label, "class_name": name, "source": "real"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    out_dir = ensure_dir(cfg["paths"]["split_dir"])
    df = collect_images(Path(cfg["paths"]["real_root"]), cfg["class_names"])
    if df.empty:
        raise RuntimeError("No images were found. Check paths.real_root and class folders.")
    train_val, test = train_test_split(df, test_size=args.test_ratio, stratify=df["label"], random_state=cfg.get("seed", 42))
    val_size = args.val_ratio / (1.0 - args.test_ratio)
    train, val = train_test_split(train_val, test_size=val_size, stratify=train_val["label"], random_state=cfg.get("seed", 42))
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)
    print(f"Saved {len(train)} train, {len(val)} val, {len(test)} test samples to {out_dir}")


if __name__ == "__main__":
    main()
