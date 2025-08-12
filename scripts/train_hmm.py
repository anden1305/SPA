"""Train a simple HMM on synthetic sleep EEG data.

Placed under scripts/, so we add the project root to sys.path to allow
imports of the sibling package `scr`. Prefer running instead with:
	python -m scr.training.train_hmm
but this shim keeps the standalone script usable.
"""

from __future__ import annotations

import sys
import pathlib

# Ensure project root (parent of scripts/) is on sys.path for `import scr`.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scr.synthetic.generator import SyntheticSleepGenerator, DEFAULT_CONFIG  
from scr.data.raw_sequence_dataset import RawEEGSequenceDataset  
from scr.models.hmm import HMM  
from scr.training.base_trainer import BaseTrainer  
import torch  
from torch.utils.data import DataLoader  


def main() -> None:
	torch.manual_seed(0)
	gen = SyntheticSleepGenerator(DEFAULT_CONFIG, seed=0)
	data = gen.generate(epochs=100)          # provides 'eeg_raw'
	ds = RawEEGSequenceDataset(data["eeg_raw"], normalize=True)
	loader = DataLoader(ds, batch_size=1)    # yields (T,1)

	model = HMM(num_states=3, obs_dim=1, normalize_time=True)
	opt = torch.optim.Adam(model.parameters(), lr=1e-3)
	# Report device
	dev = getattr(model, "device", next(model.parameters()).device)
	if dev.type == "cuda":
		print(f"Using CUDA device: {torch.cuda.get_device_name(dev)} (capability {torch.cuda.get_device_capability(dev)})")
	else:
		print(f"Using device: {dev}")

	trainer = BaseTrainer(model, opt, loader, device=dev, log_interval=1, verbose=True, log_each_step=True)
	history = trainer.fit(epochs=3)
	final = history[-1]
	print(f"Final train loss: {final['train_loss']:.4f}")


if __name__ == "__main__":  
	main()