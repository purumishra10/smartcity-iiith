from __future__ import annotations

import cv2
import numpy as np
import torch

from app.vision.model import CNN_SIZE


def bgr_to_cnn_tensor(bgr: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(bgr, (CNN_SIZE, CNN_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)
