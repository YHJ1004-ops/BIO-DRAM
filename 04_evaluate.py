from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.biodram.datasets import ThyroidImageDataset
from src.biodram.losses import BiodramCompositeLoss
from src.biodram.metrics import compute_metrics
from src.biodram.models import build_classifier
from src.biodram.transforms import build_eval_transform
from src.biodram.utils import load_config, get_device, ensure_dir


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = get_device()
    c = cfg["classifier"]
    csv_path = Path(cfg["paths"]["split_dir"]) / f"{args.split}.csv"
    ds = ThyroidImageDataset(str(csv_path), build_eval_transform(c["image_size"]))
    loader = DataLoader(ds, batch_size=c["batch_size"], shuffle=False, num_workers=c["num_workers"])
    model = build_classifier(c["architecture"], cfg["num_classes"], pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    labels = pd.read_csv(csv_path)["label"].astype(int).values
    class_counts = np.bincount(labels, minlength=cfg["num_classes"]).clip(min=1)
    cost_matrix = torch.tensor(pd.read_csv(cfg["paths"]["cost_matrix"], header=None).values, dtype=torch.float32)
    criterion = BiodramCompositeLoss(class_counts, cost_matrix, cfg).to(device)

    y_true, y_prob, paths = [], [], []
    total_loss = 0.0
    for images, targets, batch_paths in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        y_prob.append(prob)
        y_true.extend(targets.cpu().numpy().tolist())
        paths.extend(batch_paths)
        total_loss += float(loss.item()) * images.size(0)
    y_prob = np.concatenate(y_prob, axis=0)
    metrics = compute_metrics(y_true, y_prob, cfg["num_classes"])
    metrics["loss"] = total_loss / len(ds)
    print(metrics)
    out_dir = ensure_dir(Path(cfg["paths"]["output_dir"]) / "predictions")
    pred = pd.DataFrame(y_prob, columns=[f"prob_{x}" for x in cfg["class_names"]])
    pred.insert(0, "path", paths)
    pred.insert(1, "label", y_true)
    pred.insert(2, "pred", np.argmax(y_prob, axis=1))
    pred.to_csv(out_dir / f"pred_{args.split}.csv", index=False)


if __name__ == "__main__":
    main()
