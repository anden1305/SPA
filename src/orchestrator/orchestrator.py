

import json
from pathlib import Path

import torch
from src.config.config import GlobalConfig
from src.data.base_dataset import BaseDataset
from src.data.data_loader import DataLoader
from src.data.mssv_dataset import MSSVDataset
from src.data.synthetic_dataset import SyntheticDataset
from src.models.base_model import MLModel
from src.models.hmm import HMM
from src.orchestrator.train_details import TrainDetails
from src.training.trainer import Trainer
from src.validation.validator import Validator
from src.visuals.visualizer import Visualizer


class Orchestrator:
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.__set_config()
        self.__prepare()
        
    ### public methods ###
    
    def run(self):
        for i in range(self.global_config.runs):
            self.run_number = i + 1
            self.__prepare_run()
            if self.run_number == 1: #TODO refactor this line
                self.validator.validate_data()
            if self.global_config.validator.prior_validation:
                self.validator.validate()
            self.trainer.train()
            self.validator.validate()
            train_details = self.__collect_training_details()
            self.__save_info(train_details=train_details)
            self.visualizer.visualize(train_details=train_details)
            self.global_config.seed += 1
        if self.global_config.runs > 1:
            validations = self.validator.validate_runs(train_details=self.train_details)
            self.visualizer.visualize_runs(train_details=self.train_details, validations=validations)
    
    ### private methods ###
    
    def __collect_training_details(self):
        losses = self.trainer.get_losses()
        predictions = self.validator.get_predictions()
        validations = self.validator.get_validations()
        train_details = TrainDetails(
            predictions=predictions, 
            validations=validations, 
            losses=losses,
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
    
    def __prepare_run(self):
        self.dataset = self.__get_dataset()
        self.data_loader = DataLoader(dataset=self.dataset, config=self.global_config, device=self.device)
        self.model = self.__get_model(self.device)
        self.trainer = Trainer(data_loader=self.data_loader, model=self.model, config=self.global_config)
        self.validator = Validator(data_loader=self.data_loader, model=self.model, trainer=self.trainer, config=self.global_config)
        self.visualizer = Visualizer(data_loader=self.data_loader, config=self.global_config)
    
    def __prepare(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_details: list[TrainDetails] = []
        self.run_number: int = 1
        self.__make_output_dir()
        self.__save_config()

    def __make_output_dir(self):
        output_dir = Path(self.global_config.results_dir) / self.global_config.run_name / str(self.run_number)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def __save_config(self):
        with open(Path(self.global_config.results_dir) / self.global_config.run_name / "config.json", "w") as f:
            json.dump(self.global_config.model_dump(), f)

    def __get_dataset(self):
        match self.global_config.dataset.type:
            case "synthetic":
                return SyntheticDataset(config=self.global_config)
            case "mssv":
                return MSSVDataset(config=self.global_config)
            case _:
                raise ValueError(f"Unknown dataset type: {self.global_config.dataset.type}")

    def __get_model(self, device: torch.device):
        match self.global_config.model.type:
            case "hmm":
                return HMM(data_loader=self.data_loader, config=self.global_config, device=device)
            case _:
                raise ValueError(f"Unknown model type: {self.global_config.model.type}")
    
    def __str__(self):
        return (f"Orchestrator(config_path={self.config_path}, "
                f"dataset={self.dataset}, "
                f"data_loader={self.data_loader}, "
                f"model={self.model}, "
                f"trainer={self.trainer})")