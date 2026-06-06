import torch

from src.helpers.cvae_checkpoint import (
    expected_subject_emb_shape,
    is_legacy_subject_emb_checkpoint,
)


def test_legacy_subject_emb_shape():
    assert expected_subject_emb_shape(4) == (92, 4)


def test_accepts_legacy_checkpoint():
    state = {"subject_emb.weight": torch.zeros(92, 4)}
    assert is_legacy_subject_emb_checkpoint(state, 4)


def test_rejects_compact_checkpoint():
    state = {"subject_emb.weight": torch.zeros(10, 4)}
    assert not is_legacy_subject_emb_checkpoint(state, 4)
