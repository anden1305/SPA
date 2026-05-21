"""Shared pytest fixtures for the SPA test suite."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture(autouse=True)
def _set_seed():
    """Make every test deterministic."""
    torch.manual_seed(0)


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")
