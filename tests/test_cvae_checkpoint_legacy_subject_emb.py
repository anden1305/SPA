import torch

from src.helpers.cvae_checkpoint import (
    expected_subject_emb_shape,
    is_legacy_subject_emb_checkpoint,
    pad_subject_emb_to_legacy,
)


def test_legacy_subject_emb_shape():
    assert expected_subject_emb_shape(4) == (93, 4)


def test_accepts_legacy_checkpoint():
    state = {"subject_emb.weight": torch.zeros(93, 4)}
    assert is_legacy_subject_emb_checkpoint(state, 4)


def test_accepts_pre_sub092_checkpoint():
    state = {"subject_emb.weight": torch.zeros(92, 4)}
    assert is_legacy_subject_emb_checkpoint(state, 4)


def test_pad_subject_emb_to_legacy():
    state = {"subject_emb.weight": torch.ones(92, 4)}
    padded = pad_subject_emb_to_legacy(state, 4)
    assert padded["subject_emb.weight"].shape == (93, 4)
    assert torch.all(padded["subject_emb.weight"][:92] == 1)
    assert torch.all(padded["subject_emb.weight"][92] == 0)


def test_rejects_compact_checkpoint():
    state = {"subject_emb.weight": torch.zeros(10, 4)}
    assert not is_legacy_subject_emb_checkpoint(state, 4)
