import torch

EPS_END_STEP_COUNT = 1_000_000
EPS_START = 1.0
EPS_END = 0.1

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)