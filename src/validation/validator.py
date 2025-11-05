import json
import torch
import numpy as np
from typing import Any, Optional
from src.config.config import GlobalConfig
from src.data.data_loader import DataLoader
from src.data.data_loader_collection import DataLoaderCollection
from src.helpers.accuracy import accuracy
from src.helpers.align_labels import align_labels_hungarian
from src.helpers.frequency_statistics import compute_feature_statistics
from src.helpers.nmi import calculate_nmi
from src.helpers.summary_statistics import compute_summary_statistics
from src.models.base_model import BaseModel
from src.orchestrator.train_details import TrainDetails
from src.helpers.state_distinctness import compute_state_distinctness


class Validator:
    def __init__(self,
                 data_loader: DataLoaderCollection,
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
    
    def validate_epoch(self, epoch: int, optimizer: torch.optim.Optimizer):
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
            if self.config.learning_rate:
                self.validations[epoch]["learning_rate"] = optimizer.param_groups[0]['lr']
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
        has_features = self.data_loader.has_features_enabled()
        self.data_validations = {}
        out_path = f"{self.global_config.results_dir}/{self.global_config.run_name}/data_validations.json"
        if self.config.state_distinctness and has_features:
            distinctness = compute_state_distinctness(x, y)
            self.data_validations.update(distinctness)
        if self.config.summary_statistics and has_features:
            self.data_validations.update(compute_summary_statistics(x, y, self.data_loader))
            self.data_validations.update(compute_feature_statistics(x, y, self.data_loader, has_features))
            # Also compute feature-feature correlations (overall and per state)
            self.data_validations.update({
                "feature_correlations": self.__calculate_feature_correlations(x, y, has_features)
            })
        if has_features:
            separability = self.__calculate_feature_separability(x, y)
            if separability:
                self.data_validations["feature_separability"] = separability
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

    def __calculate_feature_separability(self, x: Optional[torch.Tensor], y: Optional[torch.Tensor]) -> dict[str, Any]:
        """Compute per-feature separability using a Fisher score style ratio."""
        if x is None or y is None:
            return {}

        try:
            x_np = x.detach().cpu().numpy()
        except AttributeError:
            x_np = np.asarray(x)

        try:
            y_np = y.detach().cpu().numpy().reshape(-1)
        except AttributeError:
            y_np = np.asarray(y).reshape(-1)

        if x_np.size == 0 or y_np.size == 0:
            return {}

        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)
        elif x_np.ndim > 2:
            feature_dim = x_np.shape[-1]
            x_np = x_np.reshape(-1, feature_dim)

        if x_np.shape[0] != y_np.shape[0]:
            min_len = min(x_np.shape[0], y_np.shape[0])
            x_np = x_np[:min_len]
            y_np = y_np[:min_len]

        if x_np.shape[0] == 0 or x_np.shape[1] == 0:
            return {}

        classes = np.unique(y_np)
        if classes.size < 2:
            return {}

        n_samples, n_features = x_np.shape
        global_mean = x_np.mean(axis=0)
        between_var = np.zeros(n_features, dtype=float)
        within_var = np.zeros(n_features, dtype=float)

        for cls in classes:
            mask = y_np == cls
            cls_count = int(mask.sum())
            if cls_count == 0:
                continue
            cls_data = x_np[mask]
            cls_mean = cls_data.mean(axis=0)
            if cls_count > 1:
                cls_var = cls_data.var(axis=0, ddof=1)
            else:
                cls_var = np.zeros(n_features, dtype=float)
            weight = float(cls_count)
            between_var += weight * (cls_mean - global_mean) ** 2
            within_var += weight * cls_var

        epsilon = 1e-12
        scores = between_var / (within_var + epsilon)

        feature_names = self.data_loader.get_feature_names()
        if feature_names is None or len(feature_names) != n_features:
            feature_names = [f"Feature {i + 1}" for i in range(n_features)]
        else:
            feature_names = [str(name) for name in feature_names]

        sorted_indices = np.argsort(scores)[::-1]

        return {
            "feature_names": feature_names,
            "scores": [float(score) for score in scores.tolist()],
            "between_variance": [float(val) for val in between_var.tolist()],
            "within_variance": [float(val) for val in within_var.tolist()],
            "sorted_indices": [int(idx) for idx in sorted_indices.tolist()],
            "classes": classes.tolist(),
            "sample_count": int(n_samples)
        }
    
    def __calculate_feature_correlations(self, x: Optional[torch.Tensor], y: Optional[torch.Tensor], has_features: bool) -> dict[str, Any]:
        """Compute Pearson correlation matrices between features.
        Returns a dict with overall correlation and per-state correlations.
        """
        if x is None or y is None:
            print('No data available for feature correlation calculation.')
            return {}
        
        # Ignore raw data
        if not has_features:
            print('Skipping feature correlation calculation due to high dimensionality.')
            return {}

        # Flatten to (N, D)
        try:
            x_np = x.detach().cpu().numpy()
        except AttributeError:
            x_np = np.asarray(x)

        # Limit very high dimensional inputs to keep plots readable and consistent with other stats
        if x_np.ndim == 1:
            x_np = x_np.reshape(-1, 1)
        elif x_np.ndim > 2:
            feature_dim = x_np.shape[-1]
            x_np = x_np.reshape(-1, feature_dim)

        # Truncate to the first 50 features similar to compute_feature_statistics
        if x_np.shape[1] > 50:
            x_np = x_np[:, :50]

        try:
            y_np = y.detach().cpu().numpy().reshape(-1)
        except AttributeError:
            y_np = np.asarray(y).reshape(-1)

        # Align lengths just in case
        n = min(x_np.shape[0], y_np.shape[0])
        if n <= 1 or x_np.shape[1] == 0:
            return {}
        x_np = x_np[:n]
        y_np = y_np[:n]

        # Overall correlation
        with np.errstate(invalid='ignore'):
            overall_corr = np.corrcoef(x_np, rowvar=False)

        # Per-state correlation
        per_state: dict[str, list[list[float]]] = {}
        classes = np.unique(y_np)
        for cls in classes:
            mask = (y_np == cls)
            if mask.sum() <= 1:
                # Not enough samples to compute correlation
                continue
            x_c = x_np[mask]
            with np.errstate(invalid='ignore'):
                corr_c = np.corrcoef(x_c, rowvar=False)
            per_state[str(int(cls))] = corr_c.astype(float).tolist()

        feature_names = self.data_loader.get_feature_names()
        if not feature_names or len(feature_names) != x_np.shape[1]:
            feature_names = [f"Feature {i+1}" for i in range(x_np.shape[1])]
        else:
            feature_names = [str(nm) for nm in feature_names[:x_np.shape[1]]]

        return {
            "overall": overall_corr.astype(float).tolist(),
            "per_state": per_state,
            "feature_names": feature_names
        }
    
    
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