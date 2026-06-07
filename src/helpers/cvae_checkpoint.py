"""CVAE checkpoint compatibility helpers (legacy subject indexing)."""

from __future__ import annotations

import torch

from src.data.data_loader_collection import LEGACY_SUBJECT_EMB_NUM_ROWS

SUBJECT_EMB_WEIGHT_KEY = "subject_emb.weight"


def expected_subject_emb_shape(emb_dim: int) -> tuple[int, int]:
    return (LEGACY_SUBJECT_EMB_NUM_ROWS, emb_dim)


def subject_emb_shape_from_cvae_state(cvae_state: dict[str, torch.Tensor]) -> tuple[int, ...] | None:
    weight = cvae_state.get(SUBJECT_EMB_WEIGHT_KEY)
    if weight is None:
        return None
    return tuple(weight.shape)


def is_legacy_subject_emb_checkpoint(
    cvae_state: dict[str, torch.Tensor],
    emb_dim: int,
) -> bool:
    """True when checkpoint uses legacy sub-NNN indexing ([92|93, emb_dim])."""
    shape = subject_emb_shape_from_cvae_state(cvae_state)
    if shape is None:
        return True
    rows, dim = shape
    return dim == emb_dim and rows in (92, LEGACY_SUBJECT_EMB_NUM_ROWS)


def pad_subject_emb_to_legacy(
    cvae_state: dict[str, torch.Tensor],
    emb_dim: int,
) -> dict[str, torch.Tensor]:
    """Expand pre-sub-092 checkpoints ([92, emb_dim]) to [93, emb_dim] for sub-092."""
    weight = cvae_state.get(SUBJECT_EMB_WEIGHT_KEY)
    if weight is None:
        return cvae_state
    rows, dim = weight.shape
    if rows == LEGACY_SUBJECT_EMB_NUM_ROWS:
        return cvae_state
    if rows == 92 and LEGACY_SUBJECT_EMB_NUM_ROWS == 93 and dim == emb_dim:
        pad = torch.zeros(1, dim, dtype=weight.dtype)
        return {**cvae_state, SUBJECT_EMB_WEIGHT_KEY: torch.cat([weight, pad], dim=0)}
    return cvae_state


def incompatible_subject_emb_message(
    checkpoint_shape: tuple[int, ...] | None,
    emb_dim: int,
) -> str:
    expected = expected_subject_emb_shape(emb_dim)
    if checkpoint_shape is None:
        return "checkpoint has no subject_emb.weight"
    return (
        f"checkpoint subject_emb.weight shape {list(checkpoint_shape)} "
        f"!= expected legacy {list(expected)} "
        f"(sub-NNN index → row; not compact 0..N-1)"
    )
