"""Models module for the SPA project.

This module contains different model implementations including:
- BaseModel: Abstract base class for all models
- HMM: Standard Hidden Markov Model with Gaussian emissions
- HMMAR: Autoregressive Hidden Markov Model
"""

from .base_model import BaseModel
from .hmm import HMM
from .hmm_ar import HMMAR, exponential_lags, fit_ar_ls

__all__ = ["BaseModel", "HMM", "HMMAR", "exponential_lags", "fit_ar_ls"]
