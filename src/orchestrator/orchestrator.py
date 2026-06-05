

import copy
import json
import re
from pathlib import Path
import datetime
import uuid
import cProfile
import pandas as pd
import torch
from src.config.config import GlobalConfig
from src.data.base_dataset import BaseDataset
from src.data.data_loader import DataLoader
from src.data.data_loader_collection import DataLoaderCollection
from src.data.mssv_dataset import MSSVDataset
from src.data.synthetic_dataset import SyntheticDataset
from src.models.base_model import BaseModel
from src.models.hmm import HMM
from src.models.marhmm import MARHMM
from src.models.cvae_mar_hmm import CVAEMARHMM
from src.models.vae import ConditionalVAE
from src.orchestrator.train_details import TrainDetails
from src.training.trainer import Trainer
from src.validation.validator import (
    Validator,
    record_prior_pred_nmi_on_train_details,
)
from src.helpers.profiling import write_cprofile_outputs
from src.visuals.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np


class Orchestrator:
    
    def __init__(self, config_path: str, profile: bool = False):
        self.config_path = config_path
        self.profile = profile
        self.__set_config()
        self.__prepare()
    
    ### public methods ###

    def _pretrained_checkpoint_path(self) -> Path | None:
        """Checkpoint used to seed each train_vae run (never written unless explicitly enabled)."""
        raw = self.global_config.cvae.pretrained_checkpoint_path
        if raw is None:
            raw = self.global_config.cvae.model_checkpoint_path
        return Path(raw) if raw else None

    def _remap_subject_embedding_state(self, cvae_state: dict) -> dict:
        """Map legacy subject_emb rows (sub-NNN index) onto current 0..N-1 conditioning indices."""
        key = "subject_emb.weight"
        if key not in cvae_state or not hasattr(self.model.cvae, "subject_emb"):
            return cvae_state

        old_emb = cvae_state[key]
        new_n = self.model.cvae.subject_emb.num_embeddings
        if old_emb.shape[0] == new_n:
            return cvae_state

        subject_map = self.train_loader.get_subject_map()
        new_emb = torch.zeros((new_n, old_emb.shape[1]), dtype=old_emb.dtype)
        for subject_id, new_idx in subject_map.items():
            match = re.search(r"\d+", str(subject_id))
            if not match:
                continue
            old_idx = int(match.group())
            if 0 <= old_idx < old_emb.shape[0]:
                new_emb[new_idx] = old_emb[old_idx]

        remapped = dict(cvae_state)
        remapped[key] = new_emb
        if self.global_config.verbose:
            print(
                f"Remapped subject_emb from {old_emb.shape[0]} rows (legacy indices) "
                f"to {new_n} rows (sorted dataset indices).",
                flush=True,
            )
        return remapped

    def _apply_cvae_state(self, cvae_state: dict) -> None:
        cvae_state = self._remap_subject_embedding_state(cvae_state)
        self.model.cvae.load_state_dict(cvae_state, strict=False)
        cvae = self.model.cvae
        prior = getattr(cvae, "prior", None)
        if prior in ("warm_gmm", "warm_hmm_gmm", "gmm", "hmm_gmm") and "prior_means" in cvae_state:
            cvae.gmm_warmup_initialized = True
        if prior in ("warm_hmm_gmm", "hmm_gmm") and "prior_transition_logits" in cvae_state:
            cvae.hmm_transitions_initialized = True
            cvae.hmm_transitions_initialized_at_hmm = True

    def _load_cvae_checkpoint_from(self, checkpoint_path: Path | str) -> bool:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            print(f"CVAE checkpoint not found at {checkpoint}; continuing without loading.")
            return False

        print(f"Loading CVAE model from checkpoint: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        cvae_state = {k[len("cvae."):]: v for k, v in state.items() if k.startswith("cvae.")}
        if not cvae_state:
            print(f"No cvae.* keys in checkpoint {checkpoint}; skipping load.")
            return False
        self._apply_cvae_state(cvae_state)
        return True

    def _load_cvae_checkpoint_if_available(self) -> bool:
        """Load checkpoint for validation: explicit model_checkpoint_path, else per-run dir."""
        if self.global_config.cvae.model_checkpoint_path:
            return self._load_cvae_checkpoint_from(self.global_config.cvae.model_checkpoint_path)

        if self.run_number is not None:
            run_ckpt = self._resolve_run_validation_checkpoint(
                Path(self.global_config.results_dir)
                / self.global_config.run_name
                / str(self.run_number)
                / "checkpoints"
            )
            if run_ckpt is not None:
                return self._load_cvae_checkpoint_from(run_ckpt)

        pretrained = self._pretrained_checkpoint_path()
        if pretrained is not None:
            return self._load_cvae_checkpoint_from(pretrained)
        return False

    @staticmethod
    def _resolve_run_validation_checkpoint(ckpt_dir: Path) -> Path | None:
        manifest = ckpt_dir / "validation_checkpoint.txt"
        if manifest.exists():
            path = Path(manifest.read_text().strip())
            return path if path.exists() else None
        score = ckpt_dir / "cvae_best_checkpoint_score.pth"
        if score.exists():
            return score
        best = ckpt_dir / "cvae_best_kmeans_nmi.pth"
        if best.exists():
            return best
        final = ckpt_dir / "cvae_final_model.pth"
        return final if final.exists() else None

    def _save_cvae_checkpoint_to_config_path(self):
        """Optionally mirror the latest finetuned weights to a global path (off by default)."""
        if not self.global_config.cvae.save_pretrained_checkpoint:
            return
        target = self.global_config.cvae.pretrained_checkpoint_path or self.global_config.cvae.model_checkpoint_path
        if target is None:
            return
        checkpoint = Path(target)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint)
        print(f"Saved CVAE checkpoint to configured path: {target}")
    
    def run(self):
        if self.global_config.cvae.training_pipeline == "cvae":
            raise ValueError(
                "Config uses cvae.training_pipeline='cvae'. Use --method train_vae "
                "(then validate_cvae_gmm), not --method train."
            )
        if self.global_config.verbose:
            self.__initial_print()
        if self.global_config.validate_data:
            self.validator.validate_data()
            self.visualizer.visualize_data()
        for i in range(self.global_config.runs):
            if self.global_config.verbose:
                self.__print_run()
            self.run_number = i + 1
            self.model.reset()
            self.trainer.reset()
            self.validator.validate(epoch=0)  
            run_dir = Path(self.global_config.results_dir) / self.global_config.run_name / str(self.run_number)
            if self.profile:
                self.trainer.train_profiled(run_dir, basename="train", sort="cumulative")
            else:
                self.trainer.train()
            self.validator.validate(epoch=self.trainer.current_epoch)
            train_details = self.__collect_training_details()
            self.__save_info(train_details=train_details)
            self.visualizer.visualize(train_details=train_details)
            self.global_config.seed += 1
            self.validator.reset()
        if self.global_config.runs > 1:
            if self.global_config.verbose:
                self.__print_end()
            validations = self.validator.validate_runs(train_details=self.train_details)
            self.visualizer.visualize_runs(train_details=self.train_details, validations=validations)
            

    def validate_vae_data(self):
        x, y, sub_ids = self.train_loader.get_all_data()

        stage_1 = x[y == 0]
        stage_2 = x[y == 1]
        stage_3 = x[y == 2]  # <-- keep your stages consistent; change to y==2 if needed
        # If you actually meant 3 stages: use `stage_3 = x[y == 2]`

        print("Subject IDs:", sub_ids)

        n_channels = x.shape[1]
        n_features = x.shape[-1]
        feature_idx = np.arange(n_features)

        def mean_std_per_feature(stage_tensor: torch.Tensor, channel: int):
            if stage_tensor is None or stage_tensor.numel() == 0:
                return None, None
            # shapes: (n_features,), (n_features,)
            mean = stage_tensor[:, channel, :].mean(dim=0).detach().cpu().numpy()
            std = stage_tensor[:, channel, :].std(dim=0, unbiased=False).detach().cpu().numpy()
            return mean, std

        for channel in range(3):
            m1, s1 = mean_std_per_feature(stage_1, channel)
            m2, s2 = mean_std_per_feature(stage_2, channel)
            m3, s3 = mean_std_per_feature(stage_3, channel)

            plt.figure(figsize=(12, 4))

            if m1 is not None:
                l1, = plt.plot(feature_idx, m1, label="Stage 1 (y=0)")
                plt.fill_between(feature_idx, m1 - s1, m1 + s1, alpha=0.15)

            if m2 is not None:
                l2, = plt.plot(feature_idx, m2, label="Stage 2 (y=1)")
                plt.fill_between(feature_idx, m2 - s2, m2 + s2, alpha=0.15)

            if m3 is not None:
                l3, = plt.plot(feature_idx, m3, label="Stage 3 (y=2)")
                plt.fill_between(feature_idx, m3 - s3, m3 + s3, alpha=0.15)

            plt.title(f"Channel {channel}: Mean ± Std per feature")
            plt.xlabel("Feature index")
            plt.ylabel("Value")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig('channel_{}_mean_std.png'.format(channel))
            plt.close()
    
    def validate_vae_data_magnitude(self):
        dataloaders: list[DataLoader] = self.train_loader.data_loaders
        for dataloader in dataloaders:
            name = str(dataloader.dataset)
            x, y = dataloader.get_all_data()
            # print min max for each channel in x
            n_channels = x.shape[2]
            for channel in range(n_channels):
                channel_data = x[:, :, channel].detach().cpu().numpy()
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                mean_val = np.mean(channel_data)
                std_val = np.std(channel_data)
                print(f"Dataset: {name}, Channel: {channel}, Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")

            
    
    def train_cvae(self):
        for i in range(self.global_config.runs):
            self.run_number = i + 1
            self.global_config.seed += 1
            self.model.reset()
            self.trainer.reset()
            self.validator.reset()
            # self.validator.validate(epoch=0)  
            run_dir = Path(self.global_config.results_dir) / self.global_config.run_name / str(self.run_number)
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.results_run_subdir = str(self.run_number)

            # CVAE training — always seed from read-only pretrained weights when configured.
            pretrained = self._pretrained_checkpoint_path()
            if pretrained is not None:
                self._load_cvae_checkpoint_from(pretrained)
            
            if self.global_config.cvae.training_pipeline in ['cvae_then_marhmm', 'cvae']:
                self.model.training_pipeline = 'cvae'
                self.trainer.train()
                self.validator.validate_cvae()
                try:
                    train_details = self.__collect_training_details()
                except Exception as e:
                    print(f"Error collecting training details: {e}")
                    train_details = None
                if train_details is not None:
                    self.__save_info(train_details=train_details)
                self.visualizer.visualize_cvae(
                    model=self.model, train_details=train_details, run_number=self.run_number
                )
                run_ckpt_path = ckpt_dir / "cvae_final_model.pth"
                torch.save(self.model.state_dict(), run_ckpt_path)
                self._save_cvae_checkpoint_to_config_path()
                prior_nmi: float | None = None
                try:
                    cs_enabled = self.global_config.trainer.checkpoint_score.enabled
                    score_ckpt = self.trainer.get_best_checkpoint_score_path()
                    kmeans_ckpt = self.trainer.get_best_kmeans_checkpoint_path()
                    if cs_enabled and score_ckpt is not None:
                        validation_ckpt = score_ckpt
                        best_ep = getattr(self.trainer, "_best_checkpoint_score_epoch", None)
                        print(
                            f"Validating with best checkpoint-score: {validation_ckpt} "
                            f"(epoch {(best_ep + 1) if best_ep is not None else '?'})"
                        )
                    elif kmeans_ckpt is not None:
                        validation_ckpt = kmeans_ckpt
                        best_ep = getattr(self.trainer, "_best_kmeans_epoch", None)
                        print(
                            f"Validating with best KMeans-NMI checkpoint: {validation_ckpt} "
                            f"(epoch {(best_ep + 1) if best_ep is not None else '?'})"
                        )
                    else:
                        validation_ckpt = run_ckpt_path
                        print(f"Validating with final run checkpoint: {validation_ckpt}")
                    (ckpt_dir / "validation_checkpoint.txt").write_text(str(validation_ckpt))
                    self._load_cvae_checkpoint_from(validation_ckpt)
                    prior_nmi = self.predict_cvae()
                    if train_details is not None and prior_nmi is not None:
                        record_prior_pred_nmi_on_train_details(train_details, prior_nmi)
                        self.__save_info(train_details=train_details)
                except Exception as e:
                    print(f"Warning: validation after CVAE run {self.run_number} failed: {e}")
                finally:
                    from src.validation.validator import PRIOR_PRED_NMI_KEY

                    extra: dict[str, float | int] = {}
                    if prior_nmi is not None:
                        extra["val/prior_pred_nmi"] = prior_nmi
                        extra[PRIOR_PRED_NMI_KEY] = prior_nmi
                    self.trainer.finalize_wandb(extra)
            
            if self.global_config.cvae.training_pipeline in ['marhmm', 'cvae_then_marhmm']:
                self.model.training_pipeline = 'marhmm'
                self.__prepare_run(ignore_model=True)
                self.model.data_loader = self.train_loader
                if self.global_config.cvae.reinit_marhmm:
                    self.model.reinitialize_marhmm()
                    self.validator = Validator(data_loader=self.val_loader, train_data_loader=self.train_loader, model=self.model, config=self.global_config)
                    self.trainer = Trainer(data_loader=self.train_loader, model=self.model, config=self.global_config, validator=self.validator)
                    self.visualizer = Visualizer(data_loader=self.val_loader, config=self.global_config, model=self.model, validator=self.validator)
                self.trainer.train()
                train_details = self.__collect_training_details()
                self.visualizer.visualize(train_details=train_details)
                torch.save(self.model.state_dict(), f"{self.global_config.results_dir}/{self.global_config.run_name}/cvaehmm_final_model.pth")

        self._summarize_cvae_run_metrics()
    
    
    def predict_cvae(self) -> float | None:
        """Validate CVAE with the prior's predictions (GMM marginal or HMM Viterbi).

        Returns validation NMI (same quantity written to ``plots/metrics.txt``).
        """
        loaded = self._load_cvae_checkpoint_if_available()
        if not loaded:
            raise FileNotFoundError(
                "No CVAE checkpoint available for validation. "
                "Run train_vae first, set cvae.model_checkpoint_path, or use validate_cvae_all_runs "
                "with a saved config.json (append_run_timestamp: false)."
            )
        prior = getattr(self.model.cvae, "prior", "gmm")
        if prior in ("hmm_gmm", "warm_hmm_gmm"):
            nmi, log_pz, y_hat, y, mu, x, sub_ids, switch_rate, gmm_switch = (
                self.validator.validate_cvae_hmm()
            )
            self.visualizer.visualize_cvae_hmm(
                y_hat, y, mu, x, nmi, log_pz, sub_ids, switch_rate, run_number=self.run_number
            )
            print(
                f"HMM-GMM NMI: {nmi}, log p(z_{{1:T}}): {log_pz}, "
                f"HMM switch (per 100): {switch_rate}, GMM marginal switch (per 100): {gmm_switch}"
            )
        else:
            nmi, likelihood, y_hat, y, x_latent, x, sub_ids, gmm_switch = (
                self.validator.validate_cvae_gmm()
            )
            self.visualizer.visualize_cvae_gmm(
                y_hat, y, x_latent, x, nmi, likelihood, sub_ids, run_number=self.run_number
            )
            print(f"GMM NMI: {nmi}, Likelihood: {likelihood}, switch rate (per 100): {gmm_switch}")
        return float(nmi)

    def validate_all_cvae_runs(self):
        """GMM-validate every per-run checkpoint under the current run_name (no retraining)."""
        base = Path(self.global_config.results_dir) / self.global_config.run_name
        run_dirs = sorted(
            (p for p in base.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        if not run_dirs:
            raise FileNotFoundError(f"No numbered run directories under {base}")

        summary_rows = []
        for run_dir in run_dirs:
            run_num = int(run_dir.name)
            ckpt_dir = run_dir / "checkpoints"
            validation_ckpt = self._resolve_run_validation_checkpoint(ckpt_dir)
            if validation_ckpt is None:
                print(f"Skipping run {run_num}: no checkpoint in {ckpt_dir}")
                continue
            self.run_number = run_num
            self.model.reset()
            self._load_cvae_checkpoint_from(validation_ckpt)
            self.predict_cvae()
            metrics_path = run_dir / "plots" / "metrics.txt"
            if metrics_path.exists():
                summary_rows.append({"run": run_num, "checkpoint": str(validation_ckpt), **self._parse_metrics_txt(metrics_path)})

        self._write_cvae_reliability_summary(summary_rows)

    def _summarize_cvae_run_metrics(self) -> None:
        base = Path(self.global_config.results_dir) / self.global_config.run_name
        rows = []
        for run_dir in sorted((p for p in base.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
            metrics_path = run_dir / "plots" / "metrics.txt"
            if not metrics_path.exists():
                continue
            ckpt_dir = run_dir / "checkpoints"
            validation_ckpt = self._resolve_run_validation_checkpoint(ckpt_dir)
            row = {"run": int(run_dir.name), "checkpoint": str(validation_ckpt) if validation_ckpt else ""}
            row.update(self._parse_metrics_txt(metrics_path))
            rows.append(row)
        self._write_cvae_reliability_summary(rows)

    def _write_cvae_reliability_summary(self, summary_rows: list[dict]) -> None:
        if not summary_rows:
            return
        summary_path = Path(self.global_config.results_dir) / self.global_config.run_name / "reliability_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Wrote per-run validation summary to {summary_path}")

    @staticmethod
    def _parse_metrics_txt(path: Path) -> dict:
        out: dict = {}
        for line in path.read_text().splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip().lower().replace(" ", "_")
            try:
                out[key] = float(val.strip())
            except ValueError:
                out[key] = val.strip()
        return out
    
    ### private methods ###
    
    def __collect_training_details(self):
        losses = self.trainer.get_losses()
        historic_values = self.validator.get_historic_values()
        predictions = self.validator.get_predictions()
        validations = self.validator.get_validations()
        train_details = TrainDetails(
            predictions=predictions, 
            validations=validations, 
            losses=losses,
            historic_values=historic_values,
            run_number=self.run_number,
            save_path=f"{self.global_config.results_dir}/{self.global_config.run_name}"
        )
        self.train_details.append(train_details)
        return train_details

    def __save_info(self, train_details: TrainDetails):
        train_details.save_info()
        self.model.save_info(train_details=train_details)

    def __set_config(self):
        self.global_config = GlobalConfig.from_yaml(self.config_path)
    
    def __prepare_run(self, ignore_model: bool = False):
        self.train_datasets = self.get_train_datasets()
        self.val_datasets = self.get_val_datasets()
        self.train_loader = DataLoaderCollection(datasets=self.train_datasets, config=self.global_config, device=self.device)
        self.val_loader = DataLoaderCollection(datasets=self.val_datasets, config=self.global_config, for_validation=True, device=self.device)
        if not ignore_model:
            self.model = self.__get_model(self.device)
        self.validator = Validator(data_loader=self.val_loader, train_data_loader=self.train_loader, model=self.model, config=self.global_config)
        self.trainer = Trainer(data_loader=self.train_loader, model=self.model, config=self.global_config, validator=self.validator)
        self.visualizer = Visualizer(data_loader=self.val_loader, config=self.global_config, model=self.model, validator=self.validator)

    def __prepare(self):
        if self.global_config.append_run_timestamp:
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.global_config.run_name = f"{self.global_config.run_name}_{time_str}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_details: list[TrainDetails] = []
        self.run_number: int = 1
        self.__make_output_dir()
        self.__save_config()
        self.__prepare_run()

    def __make_output_dir(self):
        output_dir = Path(self.global_config.results_dir) / self.global_config.run_name
        output_dir = output_dir / str(self.run_number)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def __save_config(self):
        with open(Path(self.global_config.results_dir) / self.global_config.run_name / "config.json", "w") as f:
            json.dump(self.global_config.model_dump(), f)

    def get_train_datasets(self):
        datasets = []
        for config in self.global_config.train_datasets:
            if config.type == "mssv" and config.lab is not None:
                datasets.extend(self.__expand_mssv_lab_configs(config))
                continue
            match config.type:
                case "synthetic":
                    datasets.append(SyntheticDataset(config=config))
                case "mssv":
                    datasets.append(MSSVDataset(config=config))
                case _:
                    raise ValueError(f"Unknown dataset type: {config.type}")
        return datasets
    
    def get_val_datasets(self):
        datasets = []
        for config in self.global_config.val_datasets:
            if config.type == "mssv" and config.lab is not None:
                datasets.extend(self.__expand_mssv_lab_configs(config))
                continue
            match config.type:
                case "synthetic":
                    datasets.append(SyntheticDataset(config=config))
                case "mssv":
                    datasets.append(MSSVDataset(config=config))
                case _:
                    raise ValueError(f"Unknown dataset type: {config.type}")
        return datasets

    def __expand_mssv_lab_configs(self, config):
        metadata_path = Path("data/ds006366_processed/metadata.csv")
        metadata = pd.read_csv(metadata_path)
        rows = metadata.loc[metadata["lab"] == config.lab]
        if rows.empty:
            raise ValueError(f"No MSSV datasets found for lab {config.lab}.")

        # Apply quality filter if specified
        if config.quality_filter is not None:
            rows = rows[rows["participant_id"].isin(config.quality_filter)]
            if rows.empty:
                raise ValueError(f"No MSSV datasets found for lab {config.lab} after quality filtering. Check quality_filter subject IDs.")

        datasets = []
        for _, row in rows.iterrows():
            datasets.append(
                MSSVDataset(
                    config=type(config)(
                        type=config.type,
                        id=str(row["participant_id"]),
                        lab=config.lab,
                        run=int(row["run"]),
                        remove_artifact=config.remove_artifact,
                        quality_filter=config.quality_filter,  # Pass through for reference
                        signals=config.signals,
                    )
                )
            )
        return datasets

    def __get_model(self, device: torch.device):
        match self.global_config.model.type:
            case "hmm":
                return HMM(data_loader=self.train_loader, config=self.global_config, device=device)
            case "marhmm":
                return MARHMM(data_loader=self.train_loader, config=self.global_config, device=device)
            case "cvae_marhmm":
                return CVAEMARHMM(data_loader=self.train_loader, config=self.global_config, device=device)
            case "cvae":
                return ConditionalVAE(data_loader=self.train_loader, config=self.global_config, device=device)
            case _:
                raise ValueError(f"Unknown model type: {self.global_config.model.type}")
    
    def __initial_print(self):
        print("\n" + "=" * 60)
        print(f"🚀 Starting: {self.global_config.run_name}")
        print("=" * 60)
        print(f"🖥️  Device:      {self.device}")
        print(f"📚 Train Dataset:     {[str(ds) for ds in self.train_datasets]}")
        print(f"📚 Val Dataset:       {[str(ds) for ds in self.val_datasets]}")
        print(f"🔄 Train Dataloader:  {self.train_loader}")
        print(f"🔄 Val Dataloader:    {self.val_loader}")
        print(f"🧠 Model:       {self.model}")
        print(f"🏋️ Trainer:      {self.trainer}")
        print("=" * 60)

    def __print_run(self):
        print("\n" + "=" * 60)
        print(f"🏃 Starting Run {self.run_number} of {self.global_config.runs}")
        print("=" * 60)  
        
    def __print_end(self):
        print("\n" + "=" * 60)
        print("✅ All runs completed successfully!")
        print("=" * 60)
        print(f"📁 Final results saved to:\n   {self.global_config.results_dir}/{self.global_config.run_name}")
        print("=" * 60 + "\n")

    def __str__(self):
        return (f"Orchestrator(config_path={self.config_path}, "
                f"train_datasets={self.train_datasets}, "
                f"train_dataloader={self.train_loader}, "
                f"val_datasets={self.val_datasets}, "
                f"val_dataloader={self.val_loader}, "
                f"model={self.model}, "
                f"trainer={self.trainer})")