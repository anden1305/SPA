"""Human-readable model family names for plots and reports."""

from __future__ import annotations

from typing import Any

from src.config.config import GlobalConfig


def model_display_name_from_dict(config: dict[str, Any]) -> str:
    model = config.get("model") or {}
    params = model.get("params") or {}
    mtype = str(model.get("type", ""))
    if mtype == "hmm":
        return "HMM"
    if mtype == "marhmm":
        return "MAR-HMM"

    prior = str(params.get("prior", "gmm"))
    emb_dim = int(params.get("emb_dim") or 0)
    cond_src = str(model.get("conditioning_source", "subject"))
    conditioned = emb_dim > 0 or cond_src in ("subject", "subject_lab")

    if prior in ("hmm_gmm", "warm_hmm_gmm"):
        return "cHMM-GMVAE" if conditioned else "HMM-GMVAE"
    return "cGMVAE" if conditioned else "GMVAE"


def model_display_name(config: GlobalConfig | dict[str, Any]) -> str:
    if isinstance(config, GlobalConfig):
        return model_display_name_from_dict(config.model_dump())
    return model_display_name_from_dict(config)
