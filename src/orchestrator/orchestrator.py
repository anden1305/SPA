

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
from src.visuals.visualizer import Visualizer


class Orchestrator:
    
    def __init__(self, config_path: str, profile: bool = False):
        self.config_path = config_path
        self.profile = profile
        self.__set_config()
        self.__prepare()
    
    ### public methods ###
    
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
    
    def train_cvae(self):
        # CVAE training
        self.trainer.train()
        self.validator.validate_cvae()
        try:
            train_details = self.__collect_training_details()
        except Exception as e:
            print(f"Error collecting training details: {e}")
            train_details = None
        self.visualizer.visualize_cvae(model=self.model, train_details=train_details)
        # MAR-HMM training
        # self.model.training_pipeline = 'marhmm'
        # self.global_config.trainer.epochs = 10000
        # self.global_config.trainer.validate_per_epoch = 10
        # self.global_config.trainer.learning_rate = 0.00005
        # self.global_config.dataloader.num_batches = 2048
        # self.global_config.dataloader.sequence_length = 32
        # self.__prepare_run(ignore_model=True)
        # self.trainer.train()
        # train_details = self.__collect_training_details()
        # self.visualizer.visualize(train_details=train_details)
    
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
        self.validator = Validator(data_loader=self.val_loader, model=self.model, config=self.global_config)
        self.trainer = Trainer(data_loader=self.train_loader, model=self.model, config=self.global_config, validator=self.validator)
        self.visualizer = Visualizer(data_loader=self.val_loader, config=self.global_config, model=self.model, validator=self.validator)

    def __prepare(self):
        time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.global_config.run_name = f"{self.global_config.run_name} [{time_str}]"
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