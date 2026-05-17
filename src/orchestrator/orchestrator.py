

import copy
import json
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
from src.validation.validator import Validator
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

    def _load_cvae_checkpoint_if_available(self) -> bool:
        """Load configured CVAE checkpoint if present. Returns True when loaded."""
        checkpoint_path = self.global_config.cvae.model_checkpoint_path
        if checkpoint_path is None:
            return False

        checkpoint = Path(checkpoint_path)
        if not checkpoint.exists():
            print(f"CVAE checkpoint not found at {checkpoint_path}; continuing without loading.")
            return False

        print(f"Loading CVAE model from checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint, map_location="cpu")
        cvae_state = {k[len("cvae."):]: v for k, v in state.items() if k.startswith("cvae.")}
        self.model.cvae.load_state_dict(cvae_state, strict=False)
        return True

    def _save_cvae_checkpoint_to_config_path(self):
        """Save the full model state to configured CVAE checkpoint path for later reuse."""
        checkpoint_path = self.global_config.cvae.model_checkpoint_path
        if checkpoint_path is None:
            return

        checkpoint = Path(checkpoint_path)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint)
        print(f"Saved CVAE checkpoint to: {checkpoint_path}")
    
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
            # self.validator.validate(epoch=0)  
            run_dir = Path(self.global_config.results_dir) / self.global_config.run_name / str(self.run_number)
        
            # CVAE training
            self._load_cvae_checkpoint_if_available()
            
            
            if self.global_config.cvae.training_pipeline in ['cvae_then_marhmm', 'cvae']:
                self.model.training_pipeline = 'cvae'
                self.trainer.train()
                self.validator.validate_cvae()
                try:
                    train_details = self.__collect_training_details()
                except Exception as e:
                    print(f"Error collecting training details: {e}")
                    train_details = None
                self.visualizer.visualize_cvae(model=self.model, train_details=train_details)
                # Save one checkpoint per run to avoid overwriting when runs > 1.
                run_ckpt_path = Path(self.global_config.results_dir) / self.global_config.run_name / f"cvae_final_model_run{self.run_number}.pth"
                torch.save(self.model.state_dict(), run_ckpt_path)
                # Also save to configured checkpoint path so follow-up validation can load it.
                self._save_cvae_checkpoint_to_config_path()
                # Also perform GMM validation using the checkpoint we just saved so
                # that each run has train -> validate ordering.
                try:
                    self.global_config.cvae.model_checkpoint_path = str(run_ckpt_path)
                    self.predict_cvae()
                except Exception as e:
                    print(f"Warning: validation after CVAE run {self.run_number} failed: {e}")
            
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
    
    
    def predict_cvae(self):
        
        # CVAE training
        loaded = self._load_cvae_checkpoint_if_available()
        if not loaded:
            raise FileNotFoundError(
                "No CVAE checkpoint available for validation. "
                "Run train_vae first or set cvae.model_checkpoint_path to an existing file."
            )
        nmi, likelihood, y_hat, y, x_latent, x, sub_ids = self.validator.validate_cvae_gmm()
        self.visualizer.visualize_cvae_gmm(y_hat, y, x_latent, x, nmi, likelihood, sub_ids)
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