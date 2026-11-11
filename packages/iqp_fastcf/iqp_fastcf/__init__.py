"""iqp_fastcf: fast, memory-lean closed-form training of mixed IQP-QCBMs on AMD GPUs (HIP, float64).

    from iqp_fastcf import train, cosine_lr, make_evaluator, make_inits, sigma_for, all_pairs
"""
from .train import train, cosine_lr, make_evaluator, LOW_MEMORY
from .inits import make_inits, sigma_for, all_pairs, walsh, INITS

__version__ = "0.2.0"
__all__ = ["train", "cosine_lr", "make_evaluator", "LOW_MEMORY", "make_inits", "sigma_for", "all_pairs", "walsh", "INITS"]
