"""
Utils - Basic Utilities
=======================
"""

import os
import random
import numpy as np
import torch
import yaml
import glob


def load_config(path: str):
    """
    Carica configurazione da file YAML.
    
    Args:
        path: Percorso file YAML
    
    Returns:
        dict: Configurazione
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """
    Imposta seed per riproducibilità.
    
    Args:
        seed: Valore seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def device():
    """
    Rileva device disponibile.
    
    Returns:
        str: 'cuda' se GPU disponibile, altrimenti 'cpu'
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_cuda(device_id: int = 0):
    """
    Configura CUDA per uso ottimale.
    
    Args:
        device_id: ID GPU da usare (default: 0)
    
    Returns:
        torch.device: Device configurato
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        device = torch.device(f"cuda:{device_id}")
        print(f"✅ CUDA enabled: {torch.cuda.get_device_name(device_id)}")
        print(f"   Device: {device}")
        print(f"   Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1e9:.1f} GB")
        
        return device
    else:
        print("⚠️  CUDA not available, using CPU")
        return torch.device("cpu")


def list_files_from_glob(glob_pattern: str):
    """
    Lista file da glob pattern (sorted).
    
    Args:
        glob_pattern: Pattern (es. "data/*.csv")
    
    Returns:
        list: Percorsi file trovati
    """
    return sorted(glob.glob(glob_pattern))


def basename_no_ext(path: str) -> str:
    """
    Estrae basename senza estensione.
    
    Args:
        path: Percorso file
    
    Returns:
        str: Nome file senza path ed estensione
    """
    return os.path.splitext(os.path.basename(path))[0]

