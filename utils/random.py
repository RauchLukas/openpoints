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
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                # warn_only needs PyTorch >= 1.12; non-strict mode would crash on
                # CrossEntropyLoss CUDA (no deterministic kernel in older builds).
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                pass
    else:
        torch.backends.cudnn.benchmark = True
