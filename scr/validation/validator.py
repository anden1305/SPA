
import json
import torch
from scr.config.config import GlobalConfig
from scr.data.data_loader import DataLoader
from scr.helpers.nmi import calculate_nmi
from scr.models.base_model import MLModel

class Validator:
    
    
    def __init__(self,
                 data_loader: DataLoader,
                 model: MLModel,
                 config: GlobalConfig):
        self.data_loader = data_loader
        self.model = model
        self.global_config = config
        self.config = self.global_config.validator
        self.validations: dict[int, dict] = {}
    
    def validate(self, epoch: int):
        self.validations[epoch] = {}
        self.model.prepare_for_inference()
        x, y = self.data_loader.get_all_data()
        with torch.no_grad():
            preds = self.model.predict(x)
            if self.config.nmi:
                nmi = calculate_nmi(preds, y)
                print(f"NMI: {nmi}")
                self.validations[epoch]["nmi"] = nmi
    
    def save_info(self):
        with open(f"{self.global_config.results_dir}/{self.global_config.run_name}/validations.json", "w") as f:
            json.dump(self.validations, f)