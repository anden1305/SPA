import json
import torch
import numpy as np
from typing import Any
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.helpers.accuracy import accuracy
from src.helpers.align_labels import align_labels_hungarian
from src.helpers.frequency_statistics import compute_frequency_statistics
from src.helpers.nmi import calculate_nmi
from src.helpers.summary_statistics import compute_summary_statistics
from src.models.base_model import BaseModel
from src.orchestrator.train_details import TrainDetails
from src.helpers.state_distinctness import compute_state_distinctness


class Validator:
    def __init__(self,
                 data_loader: DataLoader,
                 model: BaseModel,
                 config: GlobalConfig):
        self.data_loader = data_loader
        self.model = model
        self.global_config = config
        self.config = self.global_config.validator
        self.validations: dict[int, dict] = {}
        self.predictions: dict[int, list] = {}
        self.historic_values: dict[str, list] = {}
        self.data_validations: dict[str, Any] = {}

    ####### GENERAL METHODS #######

    def validate(self, epoch: int = 0):
        if not self.historic_values:
            self.historic_values = {name: [] for name, p in self.model.named_parameters() if p.requires_grad}

        self.validations[epoch] = {}
        self.model.prepare_for_inference()
        xt, yt = self.data_loader.get_all_data()
        y = yt.detach().cpu().numpy().flatten()
        with torch.no_grad():
            predst = self.model.predict(xt)
            preds = predst.detach().cpu().numpy().flatten()
            if self.config.nmi:
                nmi = calculate_nmi(preds, y)
                self.validations[epoch]["nmi"] = nmi
            if self.config.accuracy:
                try:
                    aligned_preds = align_labels_hungarian(y, preds)
                    acc = accuracy(aligned_preds, y)
                    self.validations[epoch]["accuracy"] = acc
                except Exception as e:
                    print(f"Error occurred while calculating accuracy: {e}")
                    self.validations[epoch]["accuracy"] = None
            self.predictions[epoch] = preds.tolist()
        self.__print_validation(epoch)
    
    def validate_epoch(self, epoch: int):
        """Validate model at a specific epoch during training."""          
        self.validations[epoch] = {}
        self.model.prepare_for_inference()
        xt, yt = self.data_loader.get_all_data()
        y = yt.detach().cpu().numpy().flatten()
        with torch.no_grad():
            predst = self.model.predict(xt)
            preds = predst.detach().cpu().numpy().flatten()
            if self.config.nmi:
                nmi = calculate_nmi(preds, y)
                self.validations[epoch]["nmi"] = nmi
            if self.config.accuracy:
                try:
                    aligned_preds = align_labels_hungarian(y, preds)
                    acc = accuracy(aligned_preds, y)
                    self.validations[epoch]["accuracy"] = acc
                except Exception as e:
                    print(f"Error occurred while calculating accuracy at epoch {epoch}: {e}")
                    self.validations[epoch]["accuracy"] = None
            self.predictions[epoch] = preds.tolist()
            
        # Save historic values for this epoch
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.historic_values:
                self.historic_values[name].append(p.detach().cpu().numpy())
        
        self.model.train()  # Set back to training mode
        if self.global_config.verbose:
            nmi_val = self.validations[epoch].get('nmi', 'N/A')
            acc_val = self.validations[epoch].get('accuracy', 'N/A')
            nmi_str = f"{nmi_val:.4f}" if isinstance(nmi_val, (int, float)) else str(nmi_val)
            acc_str = f"{acc_val:.4f}" if isinstance(acc_val, (int, float)) else str(acc_val)
            print(f"Epoch {epoch + 1} - NMI: {nmi_str}, Accuracy: {acc_str}")
    
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
        x, y = self.data_loader.get_all_data()
        self.data_validations = {}
        out_path = f"{self.global_config.results_dir}/{self.global_config.run_name}/data_validations.json"
        if self.config.state_distinctness:
            distinctness = compute_state_distinctness(x, y)
            self.data_validations.update(distinctness)
        if self.config.summary_statistics:
            self.data_validations.update(compute_summary_statistics(x, y, self.data_loader.dataset))
            self.data_validations.update(compute_frequency_statistics(x, y, self.data_loader))
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
    
    
    ####### PUBLIC HELPER METHODS #######
    
    def __print_validation(self, epoch: int):
        val = self.validations[epoch]

        print("\n" + "=" * 60)
        print(f"🔍 Validation Results At Epoch {epoch + 1} ".center(60, "="))
        print("=" * 60)
        if not val:
            print("No validation results available.")
        else:
            for key, value in val.items():
                if isinstance(value, float):
                    print(f"  {key:<20}: {value:>15.6f}")
                else:
                    print(f"  {key:<20}: {str(value):>15}")
        print("=" * 60 + "\n")
    
    def reset(self):
        self.validations = {}
        self.predictions = {}
        self.historic_values = {}

    def get_predictions(self) -> dict[int, list]:
        assert self.predictions, "Inference has not been run yet."
        return self.predictions
    
    def get_validations(self) -> dict[int, dict]:
        assert self.validations, "Validation has not been run yet."
        return self.validations
    
    def get_historic_values(self) -> dict[str, list]:
        assert self.historic_values, "No historic values recorded. Ensure training has been run and historic tracking is enabled in the config .yaml."
        return self.historic_values

    def get_data_validations(self) -> dict[str, Any]:
        assert self.data_validations, "Data validation has not been run yet."
        return self.data_validations