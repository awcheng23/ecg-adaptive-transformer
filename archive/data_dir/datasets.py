import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple


class ECGDataset(Dataset):
    """
    Dataset for ECG segments stored in a .npz file with:
        - 'segments': np.ndarray of shape (N, L) or (N, 1, L)
        - 'labels':   np.ndarray of shape (N,)
    """

    def __init__(self, path: str):
        super().__init__()
        
        npz = np.load(path, mmap_mode="r")

        # Load arrays
        segments = npz["segments"]          # shape (31154, 5000)
        labels   = npz["labels"]            # shape (31154,)

        # Convert to torch
        self.segments = torch.tensor(segments, dtype=torch.float32)

        # Ensure channel dimension exists → (N, 1, L)
        if self.segments.ndim == 2:
            self.segments = self.segments.unsqueeze(1)

        self.labels = torch.tensor(labels, dtype=torch.long)

        self.n = len(self.labels)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Tensor shape (1, L)
            y: Tensor shape () [scalar label]
        """
        return self.segments[idx], self.labels[idx]
