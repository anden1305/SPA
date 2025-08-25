

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
        if self.global_config.validator.prior_validation:
            self.validator.validate(epoch=0)
        self.trainer.train()
        self.trainer.save_info()
        self.validator.validate(epoch=self.global_config.trainer.epochs)
        self.validator.save_info()
        self.visualizer.visualize()
        self.model.save_info()
    
    ### private methods ###
    
    def __set_config(self):
        self.global_config = GlobalConfig.from_yaml(self.config_path)
    
    def __prepare(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = self.__get_dataset()
        self.data_loader = self.__get_dataloader(self.dataset, self.device)
        self.model = self.__get_model(self.device)
        self.trainer = self.__get_trainer(self.data_loader, self.model)
        self.validator = Validator(data_loader=self.data_loader, model=self.model, config=self.global_config)
        self.visualizer = Visualizer(data_loader=self.data_loader, model=self.model, trainer=self.trainer, config=self.global_config, validator=self.validator)
        self.__make_output_dir()
        self.__save_config()

    def __make_output_dir(self):
        output_dir = Path(self.global_config.results_dir) / self.global_config.run_name
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def __save_config(self):
        with open(Path(self.global_config.results_dir) / self.global_config.run_name / "config.json", "w") as f:
            json.dump(self.global_config.model_dump(), f)

    def __get_trainer(self, 
                      data_loader: DataLoader,
                      model: MLModel):
        return Trainer(
            data_loader=data_loader,
            model=model,
            config=self.global_config
        )

    def __get_dataloader(self, dataset: BaseDataset, device: torch.device):
        return DataLoader(dataset=dataset, config=self.global_config, device=device)

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