import numpy as np
import torch

def _as_float32_contiguous(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float32)

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def set_confidence(avg_prolob: float, no_speech_prob: float) -> float:
    p = float(np.exp(min(0.0, float(avg_prolob))))
    conf = p * (1.0 - float(no_speech_prob))

    return max(0.0, min(1.0, conf))
