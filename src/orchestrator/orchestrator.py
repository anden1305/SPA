

import copy
import json
from pathlib import Path
import datetime
import uuid
import cProfile
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
from src.validation.validator import Validator
from src.helpers.profiling import write_cprofile_outputs
from src.helpers.cvae_checkpoint import (
    expected_subject_emb_shape,
    incompatible_subject_emb_message,
    is_legacy_subject_emb_checkpoint,
    pad_subject_emb_to_legacy,
    subject_emb_shape_from_cvae_state,
)
from src.visuals.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np


class Orchestrator:
    
    def __init__(self, config_path: str, profile: bool = False, mode: str = "train", validation_tag: str | None = None):
        self.config_path = config_path
        self.profile = profile
        self.mode = mode
        self.validation_tag = validation_tag
        self.__set_config()
        if self.mode == "validate_cvae_gmm":
            self.__prepare_for_validation()
        else:
            self.__prepare()
    
    ### public methods ###

    def _load_cvae_checkpoint_from(self, checkpoint_path: Path | str) -> bool:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            print(f"CVAE checkpoint not found at {checkpoint}; continuing without loading.")
            return False

        state = torch.load(checkpoint, map_location="cpu")
        cvae_state = {k[len("cvae."):]: v for k, v in state.items() if k.startswith("cvae.")}
        if not cvae_state:
            print(f"No cvae.* keys in checkpoint {checkpoint}; skipping load.")
            return False

        emb_dim = int(getattr(self.model.cvae, "emb_dim", 0) or 0)
        if emb_dim > 0 and not is_legacy_subject_emb_checkpoint(cvae_state, emb_dim):
            ckpt_shape = subject_emb_shape_from_cvae_state(cvae_state)
            print(
                "Skipping CVAE checkpoint load — incompatible subject embedding. "
                f"{incompatible_subject_emb_message(ckpt_shape, emb_dim)}. "
                f"Expected {list(expected_subject_emb_shape(emb_dim))}. "
                "Training from scratch; a compatible checkpoint will be saved at run end."
            )
            return False

        cvae_state = pad_subject_emb_to_legacy(cvae_state, emb_dim)
        print(f"Loading CVAE model from checkpoint: {checkpoint}")
        self.model.cvae.load_state_dict(cvae_state, strict=False)
        prior = getattr(self.model.cvae, "prior", None)
        if prior in ("warm_gmm", "warm_hmm_gmm", "gmm", "hmm_gmm") and "prior_means" in cvae_state:
            self.model.cvae.gmm_warmup_initialized = True
        if prior in ("warm_hmm_gmm", "hmm_gmm") and "prior_transition_logits" in cvae_state:
            self.model.cvae.hmm_transitions_initialized = True
            self.model.cvae.hmm_transitions_initialized_at_hmm = True
        return True

    @staticmethod
    def _resolve_run_validation_checkpoint(ckpt_dir: Path) -> Path | None:
        manifest = ckpt_dir / "validation_checkpoint.txt"
        if manifest.exists():
            path = Path(manifest.read_text().strip())
            return path if path.exists() else None
        score = ckpt_dir / "cvae_best_checkpoint_score.pth"
        if score.exists():
            return score
        best = ckpt_dir / "cvae_best_prior_pred_nmi.pth"
        if best.exists():
            return best
        final = ckpt_dir / "cvae_final_model.pth"
        return final if final.exists() else None

    def _load_cvae_checkpoint_if_available(self) -> bool:
        """Load configured CVAE checkpoint, else per-run best/final under checkpoints/."""
        if self.global_config.cvae.model_checkpoint_path:
            return self._load_cvae_checkpoint_from(self.global_config.cvae.model_checkpoint_path)

        if self.run_number is not None:
            ckpt_dir = (
                Path(self.global_config.results_dir)
                / self.global_config.run_name
                / str(self.run_number)
                / "checkpoints"
            )
            run_ckpt = self._resolve_run_validation_checkpoint(ckpt_dir)
            if run_ckpt is not None:
                return self._load_cvae_checkpoint_from(run_ckpt)
        return False

    def _save_cvae_checkpoint_to_config_path(self):
        """Optionally mirror the latest finetuned weights to model_checkpoint_path."""
        if not self.global_config.cvae.save_pretrained_checkpoint:
            return
        checkpoint_path = self.global_config.cvae.model_checkpoint_path
        if checkpoint_path is None:
            return

        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint)
        print(f"Saved CVAE checkpoint to: {checkpoint_path}")

    def _resolve_validation_run_name(self) -> str | None:
        results_dir = Path(self.global_config.results_dir)
        base_name = self.global_config.run_name
        direct_dir = results_dir / base_name
        if direct_dir.exists():
            return base_name
        candidates = [d for d in results_dir.glob(f"{base_name}_*") if d.is_dir()]
        if not candidates:
            return None

        def parse_timestamp(dir_path: Path) -> datetime.datetime | None:
            suffix = dir_path.name[len(base_name) + 1:]
            try:
                return datetime.datetime.strptime(suffix, "%Y%m%d-%H%M%S")
            except ValueError:
                return None

        candidates_with_ts = [(d, parse_timestamp(d)) for d in candidates]
        with_ts = [pair for pair in candidates_with_ts if pair[1] is not None]
        if with_ts:
            return max(with_ts, key=lambda pair: pair[1])[0].name
        return max(candidates, key=lambda d: d.stat().st_mtime).name

    def _match_checkpoint_to_run(self, checkpoint: Path, run_dir: Path) -> str | None:
        if not checkpoint.exists() or not run_dir.exists():
            return None
        candidates = list(run_dir.glob("cvae_final_model_run*.pth"))
        if not candidates:
            return None
        try:
            checkpoint_size = checkpoint.stat().st_size
        except OSError:
            return None
        size_matches = [c for c in candidates if c.stat().st_size == checkpoint_size]
        if len(size_matches) == 1:
            return size_matches[0].name
        if len(size_matches) > 1:
            return max(size_matches, key=lambda c: c.stat().st_mtime).name
        return None

    def _write_validation_info(
        self,
        nmi: float | None = None,
        likelihood: float | None = None,
        output_subdir: str | None = None,
    ):
        run_dir = Path(self.global_config.results_dir) / self.global_config.run_name
        info_dir = run_dir
        if output_subdir:
            info_dir = run_dir / "plots" / output_subdir
        info_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.global_config.cvae.model_checkpoint_path
        info: dict[str, object] = {
            "results_dir": str(self.global_config.results_dir),
            "run_name": self.global_config.run_name,
            "model_type": self.global_config.model.type,
            "checkpoint_path": checkpoint_path,
            "validation_timestamp": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            "validation_tag": output_subdir,
        }
        if checkpoint_path:
            checkpoint = Path(checkpoint_path)
            if checkpoint.exists():
                stat = checkpoint.stat()
                info["checkpoint_size"] = stat.st_size
                info["checkpoint_mtime"] = stat.st_mtime
                matched = self._match_checkpoint_to_run(checkpoint, run_dir)
                if matched:
                    info["matched_run_checkpoint"] = matched
        if nmi is not None:
            info["nmi"] = float(nmi)
        if likelihood is not None:
            info["likelihood"] = float(likelihood)

        with open(info_dir / "validation_info.json", "w") as f:
            json.dump(info, f, indent=2)
    
    def run(self):
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
                self.trainer.train_profiled(run_dir, basename="train")
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
            run_dir = Path(self.global_config.results_dir) / self.global_config.run_name / str(self.run_number)
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir = run_dir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            self.trainer.results_run_subdir = str(self.run_number)

            if self.global_config.cvae.model_checkpoint_path:
                self._load_cvae_checkpoint_from(self.global_config.cvae.model_checkpoint_path)

            if self.global_config.cvae.traning_pipeline in ['cvae_then_marhmm', 'cvae']:
                self.model.training_pipeline = 'cvae'
                if self.profile:
                    self.trainer.train_profiled(run_dir, basename="train")
                else:
                    self.trainer.train()
                self.validator.validate_cvae()
                try:
                    train_details = self.__collect_training_details()
                except Exception as e:
                    print(f"Error collecting training details: {e}")
                    train_details = None
                if train_details is not None:
                    self.__save_info(train_details=train_details)
                self.visualizer.visualize_cvae(model=self.model, train_details=train_details)

                run_ckpt_path = ckpt_dir / "cvae_final_model.pth"
                torch.save(self.model.state_dict(), run_ckpt_path)
                legacy_ckpt = (
                    Path(self.global_config.results_dir)
                    / self.global_config.run_name
                    / f"cvae_final_model_run{self.run_number}.pth"
                )
                torch.save(self.model.state_dict(), legacy_ckpt)
                self._save_cvae_checkpoint_to_config_path()

                prior_nmi: float | None = None
                try:
                    cs_enabled = self.global_config.trainer.checkpoint_score.enabled
                    score_ckpt = self.trainer.get_best_checkpoint_score_path()
                    prior_ckpt = self.trainer.get_best_prior_checkpoint_path()
                    if cs_enabled and score_ckpt is not None:
                        validation_ckpt = score_ckpt
                        best_ep = self.trainer._best_checkpoint_score_epoch
                        print(
                            f"Validating with best checkpoint-score: {validation_ckpt} "
                            f"(epoch {(best_ep + 1) if best_ep is not None else '?'})"
                        )
                    elif prior_ckpt is not None:
                        validation_ckpt = prior_ckpt
                        best_ep = self.trainer._best_prior_epoch
                        print(
                            f"Validating with best prior-pred NMI checkpoint: {validation_ckpt} "
                            f"(epoch {(best_ep + 1) if best_ep is not None else '?'})"
                        )
                    else:
                        validation_ckpt = run_ckpt_path
                        print(f"Validating with final run checkpoint: {validation_ckpt}")
                    (ckpt_dir / "validation_checkpoint.txt").write_text(str(validation_ckpt))
                    self._load_cvae_checkpoint_from(validation_ckpt)
                    prior_nmi = self._validate_trained_cvae_prior(run_dir)
                except Exception as e:
                    print(f"Warning: post-train prior validation failed: {e}")
                finally:
                    from src.validation.validator import PRIOR_PRED_NMI_KEY

                    extra: dict[str, float | int] = {}
                    if prior_nmi is not None:
                        extra["val/prior_pred_nmi"] = prior_nmi
                        extra[PRIOR_PRED_NMI_KEY] = prior_nmi
                    self.trainer.finalize_wandb(extra)
            
            if self.global_config.cvae.traning_pipeline in ['marhmm', 'cvae_then_marhmm']:
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
    
    
    def _validate_trained_cvae_prior(self, run_dir: Path) -> float:
        """Run prior-based validation on in-memory weights (best or final checkpoint)."""
        plots_dir = run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        prior = getattr(self.model.cvae, "prior", "gmm")
        if prior in ("hmm_gmm", "warm_hmm_gmm"):
            nmi, log_pz, y_hat, y, mu, x, sub_ids, switch_rate, gmm_switch = self.validator.validate_cvae_hmm()
            self.visualizer.visualize_cvae_hmm(y_hat, y, mu, x, nmi, log_pz, sub_ids, switch_rate, output_subdir=str(run_dir.name))
            self._write_validation_info(nmi=nmi, likelihood=float(log_pz), output_subdir=str(run_dir.name))
            print(f"HMM-GMM NMI: {nmi}, log p(z_1:T): {log_pz}, HMM switch (per 100): {switch_rate}, GMM marginal switch (per 100): {gmm_switch}")
        else:
            nmi, likelihood, y_hat, y, x_latent, x, sub_ids = self.validator.validate_cvae_gmm()
            self.visualizer.visualize_cvae_gmm(
                y_hat, y, x_latent, x, nmi, likelihood, sub_ids, output_subdir=str(run_dir.name)
            )
            self._write_validation_info(nmi=nmi, likelihood=likelihood, output_subdir=str(run_dir.name))
            print(f"GMM NMI: {nmi}, Likelihood: {likelihood}")
        return float(nmi)

    def predict_cvae(self):

        # CVAE training
        loaded = self._load_cvae_checkpoint_if_available()
        if not loaded:
            raise FileNotFoundError(
                "No CVAE checkpoint available for validation. "
                "Run train_vae first or set cvae.model_checkpoint_path to an existing file."
            )
        prior = getattr(self.model.cvae, "prior", "gmm")
        if prior in ("hmm_gmm", "warm_hmm_gmm"):
            nmi, log_pz, y_hat, y, mu, x, sub_ids, switch_rate, gmm_switch = self.validator.validate_cvae_hmm()
            self.visualizer.visualize_cvae_hmm(y_hat, y, mu, x, nmi, log_pz, sub_ids, switch_rate, output_subdir=self.validation_tag)
            self._write_validation_info(nmi=nmi, likelihood=float(log_pz), output_subdir=self.validation_tag)
            print(f"HMM-GMM NMI: {nmi}, log p(z_1:T): {log_pz}, HMM switch (per 100): {switch_rate}, GMM marginal switch (per 100): {gmm_switch}")
        else:
            nmi, likelihood, y_hat, y, x_latent, x, sub_ids = self.validator.validate_cvae_gmm()
            self.visualizer.visualize_cvae_gmm(y_hat, y, x_latent, x, nmi, likelihood, sub_ids, output_subdir=self.validation_tag)
            self._write_validation_info(nmi=nmi, likelihood=likelihood, output_subdir=self.validation_tag)
            print(f"GMM NMI: {nmi}, Likelihood: {likelihood}")
        
    
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
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.global_config.run_name = f"{self.global_config.run_name}_{time_str}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_details: list[TrainDetails] = []
        self.run_number: int = 1
        self.__make_output_dir()
        self.__save_config()
        self.__prepare_run()

    def __prepare_for_validation(self):
        resolved_name = self._resolve_validation_run_name()
        if resolved_name:
            self.global_config.run_name = resolved_name
        else:
            time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.global_config.run_name = f"{self.global_config.run_name}_{time_str}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_details: list[TrainDetails] = []
        if self.validation_tag and str(self.validation_tag).isdigit():
            self.run_number = int(self.validation_tag)
        else:
            self.run_number = 1
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
            match config.type:
                case "synthetic":
                    datasets.append(SyntheticDataset(config=config))
                case "mssv":
                    datasets.append(MSSVDataset(config=config))
                case _:
                    raise ValueError(f"Unknown dataset type: {config.type}")
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