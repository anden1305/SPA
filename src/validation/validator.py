import json
import torch
from typing import Any
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.helpers.nmi import calculate_nmi
from src.helpers.summary_statistics import compute_summary_statistics
from src.models.base_model import MLModel
from src.orchestrator.train_details import TrainDetails
from src.training.trainer import Trainer
from src.helpers.state_distinctness import compute_state_distinctness

class Validator:
    def __init__(self,
                 data_loader: DataLoader,
                 model: MLModel,
                 trainer: Trainer,
                 config: GlobalConfig):
        self.data_loader = data_loader
        self.model = model
        self.trainer = trainer
        self.global_config = config
        self.config = self.global_config.validator
        self.validations: dict[int, dict] = {}
        self.predictions: dict[int, list] = {}
        self.data_validations: dict[int, dict] = {}

    ####### GENERAL METHODS #######

    def validate(self):
        epoch = self.trainer.current_epoch
        self.validations[epoch] = {}
        self.model.prepare_for_inference()
        x, y = self.data_loader.get_all_data(shuffle=False)
        with torch.no_grad():
            preds = self.model.predict(x)
            if self.config.nmi:
                nmi = calculate_nmi(preds, y)
                print(f"NMI: {nmi}")
                self.validations[epoch]["nmi"] = nmi
            self.predictions[epoch] = preds.tolist()
    
    def validate_runs(self, train_details: list[TrainDetails]):
        validations = {}
        if self.config.nmi:
            nmi = {details.run_number: details.get_trained_validations()['nmi'] for details in train_details}
            validations['nmi'] = nmi
        if self.config.cross_nmi:
            cross_nmi = self.__calculate_cross_nmi(train_details)
            validations['cross_nmi'] = cross_nmi
        losses = [list(detail.losses.values()) for detail in train_details]
        final_losses = [loss[-1] for loss in losses]
        validations['loss'] = {details.run_number: final_losses[i] for i, details in enumerate(train_details)}
        with open(f"{self.global_config.results_dir}/{self.global_config.run_name}/validations.json", "w") as f:
            json.dump(validations, f)
        return validations

    def validate_data(self):
        x, y = self.data_loader.get_all_data(shuffle=False)
        self.data_validations = {}
        out_path = f"{self.global_config.results_dir}/{self.global_config.run_name}/data_validations.json"
        if self.config.state_distinctness:
            distinctness = compute_state_distinctness(x, y)
            self.data_validations.update(distinctness)
        if self.config.summary_statistics:
            self.data_validations.update(compute_summary_statistics(x, y))
        with open(out_path, "w") as f:
            json.dump(self.data_validations, f, indent=2)




    ####### HELPER METHODS #######

    def __calculate_cross_nmi(self, train_details: list[TrainDetails]):
        cross_nmis: dict[int, float] = {}
        for details in train_details:
            nmis = []
            predictions = details.get_trained_predictions()
            for other_details in train_details:
                if details != other_details:
                    nmi = calculate_nmi(predictions, other_details.get_trained_predictions())
                    nmis.append(nmi)
            cross_nmis[details.run_number] = sum(nmis) / len(nmis)
        return cross_nmis
    
    
    ####### GETTER METHODS #######

    def get_predictions(self) -> dict[int, list]:
        assert self.predictions, "Inference has not been run yet."
        return self.predictions
    
    def get_validations(self) -> dict[int, dict]:
        assert self.validations, "Validation has not been run yet."
        return self.validations
    
    def get_data_validations(self) -> dict[str, Any]:
        assert self.data_validations, "Data validation has not been run yet."
        return self.data_validations