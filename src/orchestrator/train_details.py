from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

@dataclass(slots=True, frozen=False)
class TrainDetails:
    predictions: dict[int, list]
    validations: dict[int, dict]
    losses: dict[int, float]
    run_number: int
    save_path: str
    param_history: dict[str, list] = field(default_factory=dict)
    
    def save_info(self):
        path = self.get_path()
        path.mkdir(parents=True, exist_ok=True)
        with open(f"{path}/losses.json", "w") as f:
            json.dump(self.losses, f)
        with open(f"{path}/validations.json", "w") as f:
            json.dump(self.validations, f)
        with open(f"{path}/predictions.json", "w") as f:
            json.dump(self.predictions, f)
        with open(f"{path}/param_history.json", "w") as f:
            json.dump({k: [np.asarray(v).tolist() for v in vals] if isinstance(vals, list) else []
                           for k, vals in self.param_history.items()}, f)

    def get_path(self):
        return Path(f"{self.save_path}/{self.run_number}")
    
    def get_trained_predictions(self):
        return np.asarray(self.predictions[max(self.predictions.keys())]).reshape(-1)

    def get_trained_validations(self):
        return self.validations[max(self.validations.keys())]

    def get_initial_predictions(self):
        return np.asarray(self.predictions[min(self.predictions.keys())]).reshape(-1)
    
    def get_initial_validations(self):
        return self.validations[min(self.validations.keys())]
