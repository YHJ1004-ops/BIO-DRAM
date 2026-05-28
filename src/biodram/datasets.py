from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ThyroidImageDataset(Dataset):
    def __init__(self, csv_path: str, transform: Optional[Callable] = None):
        self.df = pd.read_csv(csv_path)
        required = {"path", "label"}
        missing = required.difference(set(self.df.columns))
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = Path(row["path"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)
