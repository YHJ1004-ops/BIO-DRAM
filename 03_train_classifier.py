from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.biodram.datasets import ThyroidImageDataset
from src.biodram.losses import BiodramCompositeLoss
from src.biodram.metrics import compute_metrics
from src.biodram.models import build_classifier
from src.biodram.transforms import build_train_transform, build_eval_transform
from src.biodram.utils import load_config, set_seed, ensure_dir, get_device


def append_synthetic(train_csv: Path, synthetic_root: Path, class_names):
    df = pd.read_csv(train_csv)
    rows = []
    for label, name in enumerate(class_names):
        folder = synthetic_root / name
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
                rows.append({"path": str(p.resolve()), "label": label, "class_name": name, "source": "synthetic"})
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df


def run_epoch(model, loader, criterion, optimizer, device, amp_enabled):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    total_loss = 0.0
    for images, targets, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    y_true, y_prob = [], []
    for images, targets, _ in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        y_prob.append(prob)
        y_true.extend(targets.cpu().numpy().tolist())
        total_loss += float(loss.item()) * images.size(0)
    y_prob = np.concatenate(y_prob, axis=0)
    metrics = compute_metrics(y_true, y_prob, num_classes)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--quick_epochs", type=int, default=None)
    parser.add_argument("--output_suffix", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    out_dir = ensure_dir(Path(cfg["paths"]["output_dir"]))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    split_dir = Path(cfg["paths"]["split_dir"])

    temp_train_csv = out_dir / f"train_with_synthetic{args.output_suffix}.csv"
    train_df = append_synthetic(split_dir / "train.csv", Path(cfg["paths"]["synthetic_root"]), cfg["class_names"])
    train_df.to_csv(temp_train_csv, index=False)

    c = cfg["classifier"]
    train_ds = ThyroidImageDataset(str(temp_train_csv), build_train_transform(c["image_size"]))
    val_ds = ThyroidImageDataset(str(split_dir / "val.csv"), build_eval_transform(c["image_size"]))
    train_loader = DataLoader(train_ds, batch_size=c["batch_size"], shuffle=True, num_workers=c["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=c["batch_size"], shuffle=False, num_workers=c["num_workers"], pin_memory=True)

    labels = pd.read_csv(temp_train_csv)["label"].astype(int).values
    class_counts = np.bincount(labels, minlength=cfg["num_classes"]).clip(min=1)
    cost_matrix = torch.tensor(pd.read_csv(cfg["paths"]["cost_matrix"], header=None).values, dtype=torch.float32)

    model = build_classifier(c["architecture"], cfg["num_classes"], c.get("pretrained", True)).to(device)
    criterion = BiodramCompositeLoss(class_counts, cost_matrix, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(c["epochs"])))

    epochs = args.quick_epochs or int(c["epochs"])
    best_auc = -1.0
    patience = 0
    best_path = ckpt_dir / f"best{args.output_suffix}.pt"
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, bool(c.get("amp", True) and device.type == "cuda"))
        val_metrics = evaluate_model(model, val_loader, criterion, device, cfg["num_classes"])
        scheduler.step()
        current = val_metrics.get("macro_auc", 0.0)
        if np.isnan(current):
            current = val_metrics.get("macro_f1", 0.0)
        print({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        if current > best_auc:
            best_auc = current
            patience = 0
            torch.save({"model": model.state_dict(), "config": cfg, "metrics": val_metrics}, best_path)
        else:
            patience += 1
            if patience >= int(c.get("early_stop_patience", 12)):
                break
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
