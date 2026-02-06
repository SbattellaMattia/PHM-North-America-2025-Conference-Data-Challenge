import os
import random
import numpy as np
import torch
import yaml

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def list_files_from_glob(glob_pattern: str):
    import glob
    return sorted(glob.glob(glob_pattern))

def basename_no_ext(path: str) -> str:
    import os
    return os.path.splitext(os.path.basename(path))[0]
