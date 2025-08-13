"""Minimal HMM training on synthetic sleep EEG.

Usage (standalone):
	python scripts/train_hmm.py --epochs 5 --num-states 4
Or module form (preferred):
	python -m scripts.train_hmm --epochs 5

Focus: brevity + clarity. Supports seeding, val split, save path.
"""

from __future__ import annotations

import sys, pathlib, argparse, random
import numpy as np
import torch

# Ensure project root (parent of scripts/) is on sys.path for `import scr`.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from scr.synthetic.generator import SyntheticSleepGenerator, DEFAULT_CONFIG
from scr.data.raw_sequence_dataset import RawEEGSequenceDataset
from scr.models.hmm import HMM
from scr.training.base_trainer import BaseTrainer
from torch.utils.data import DataLoader, random_split


def parse_args():
	P = argparse.ArgumentParser(description="Train HMM on synthetic EEG")
	P.add_argument("--epochs", type=int, default=5)
	P.add_argument("--num-states", type=int, default=3)
	P.add_argument("--lr", type=float, default=1e-3)
	P.add_argument("--batch-size", type=int, default=1)
	P.add_argument("--val-split", type=float, default=0.0, help="Fraction for validation")
	P.add_argument("--seed", type=int, default=0)
	P.add_argument("--save", type=pathlib.Path, default=None, help="File to save model (.pt)")
	return P.parse_args()


def seed_all(seed: int):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def main():
	args = parse_args()
	seed_all(args.seed)

	gen = SyntheticSleepGenerator(DEFAULT_CONFIG, seed=args.seed)
	data = gen.generate(epochs=100)  # contains 'eeg_raw'
	ds = RawEEGSequenceDataset(data["eeg_raw"], normalize=True)

	if 0 < args.val_split < 0.5 and len(ds) > 1:
		val_len = max(1, int(len(ds) * args.val_split))
		train_len = len(ds) - val_len
		train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
		val_loader = DataLoader(val_ds, batch_size=args.batch_size)
	else:
		train_ds, val_loader = ds, None

	train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=len(train_ds) > 1)

	model = HMM(num_states=args.num_states, obs_dim=1, normalize_time=True)
	opt = torch.optim.Adam(model.parameters(), lr=args.lr)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")

	trainer = BaseTrainer(model, opt, train_loader, val_loader=val_loader, device=device, log_interval=10, verbose=True)
	history = trainer.fit(epochs=args.epochs)
	final = history[-1]
	print(f"Final train loss: {final['train_loss']:.4f}")
	if val_loader and 'val_loss' in final:
		print(f"Final val loss: {final['val_loss']:.4f}")
	if args.save:
		args.save.parent.mkdir(parents=True, exist_ok=True)
		torch.save({"model_state": model.state_dict(), "args": vars(args)}, args.save)
		print(f"Saved model -> {args.save}")


if __name__ == "__main__":
	main()