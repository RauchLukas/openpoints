import os
import random

import numpy as np
import torch

# Required for deterministic cuBLAS matmul on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_random_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Do not call torch.use_deterministic_algorithms(): CrossEntropyLoss has no
        # deterministic CUDA kernel on PyTorch 1.x / CUDA builds used here.
        # cudnn deterministic + fixed seeds are sufficient for val reproducibility.
    else:
        torch.backends.cudnn.benchmark = True
