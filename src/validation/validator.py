import json
import torch
import numpy as np
from typing import Any, Optional
from src.config.config import GlobalConfig
from src.data.data_loader_collection import DataLoaderCollection
from src.helpers.accuracy import accuracy
from src.helpers.align_labels import align_labels_hungarian
from src.helpers.frequency_statistics import compute_feature_statistics
from src.helpers.metrics_aggregation import summarize_metrics
from src.helpers.nmi import calculate_nmi
from src.helpers.summary_statistics import compute_summary_statistics
from src.models.base_model import BaseModel
from src.models.cvae_mar_hmm import CVAEMARHMM
from src.orchestrator.train_details import TrainDetails
from src.helpers.state_distinctness import compute_state_distinctness
from src.validation.hmmgmm_metrics import (
    checkpoint_score,
    entropy_norm,
    latent_autocorr,
    state_switch_rate,
)

# Post-train prior predictions (GMM marginal or HMM Viterbi) — paper / sweep objective.
PRIOR_PRED_NMI_KEY = "prior_pred_nmi"
PRIOR_PRED_VALIDATION_EPOCH = 10**9
PRIOR_PRED_NMI_SWEEP_METRIC = "val/prior_pred_nmi"


def record_prior_pred_nmi_on_train_details(train_details: Any, nmi: float) -> None:
    """Attach post-train prior-prediction NMI for sweep aggregation and validations.json."""
    train_details.validations[PRIOR_PRED_VALIDATION_EPOCH] = {
        PRIOR_PRED_NMI_KEY: float(nmi),
    }


class Validator:
    def __init__(self,
                 data_loader: DataLoaderCollection,
                 train_data_loader: DataLoaderCollection,
                 model: BaseModel,
                 config: GlobalConfig):
        self.data_loader = data_loader
        self.train_data_loader = train_data_loader
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
        xt, yt, sub_ids = self.data_loader.get_all_data()
        with torch.no_grad():
            if type(self.model) == CVAEMARHMM:
                if (
                    getattr(self.model, "training_pipeline", None) == "cvae"
                    or self.global_config.cvae.training_pipeline == "cvae"
                ):
                    self.validate_cvae_epoch(epoch)
                    return
                predst = self.model.predict(xt, sub_ids)
            else:
                predst = self.model.predict(xt)
            
            # Handle MARHMM burn-in
            if hasattr(self.model, 'max_lag') and self.model.max_lag > 0:
                predst = predst[:, self.model.max_lag:]
                yt = yt[:, self.model.max_lag:]
            
            y = yt.detach().cpu().numpy().flatten()
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
        xt, yt, sub_ids = self.data_loader.get_all_data()
        with torch.no_grad():
            if self.config.log_likelihood:
                if type(self.model) == CVAEMARHMM:
                    val_nll, _ = self.model.forward(xt, sub_ids, epoch)
                else:
                    val_nll = self.model.forward(xt)
                
                self.validations[epoch]['log_likelihood'] = -val_nll.item()
                self.validations[epoch]['likelihood'] = np.exp(-val_nll.item())
                self.validations[epoch]['nll'] = val_nll.item()
            if type(self.model) == CVAEMARHMM:
                predst = self.model.predict(xt, sub_ids)
            else:
                predst = self.model.predict(xt)
            
            # Handle MARHMM burn-in
            if hasattr(self.model, 'max_lag') and self.model.max_lag > 0:
                predst = predst[:, self.model.max_lag:]
                yt = yt[:, self.model.max_lag:]
            
            y = yt.detach().cpu().numpy().flatten()
            preds = predst.detach().cpu().numpy().flatten()
            if self.config.nmi:
                nmi = calculate_nmi(preds, y)
                self.validations[epoch]["nmi"] = nmi
                
        # Compute state entropy and perplexity for validation data
        entropy_metrics = self.__compute_state_entropy(preds, self.data_loader.get_num_states())
        self.validations[epoch]["entropy"] = entropy_metrics["entropy"]
        self.validations[epoch]["perplexity"] = entropy_metrics["perplexity"]
        
        perp = entropy_metrics["perplexity"] / self.train_data_loader.get_num_states()
        ll = self.validations[epoch]["likelihood"]
        eps = 1e-12
        w = 0.9
        nlpp = w * np.log(ll + eps) + (1 - w) * np.log(perp + eps)
        self.validations[epoch]["nlpp"] = nlpp
        
        # Compute train NMI if train data loader is available
        if self.config.validate_train and self.config.nmi and self.train_data_loader is not None:
            x_train, y_train, sub_ids = self.train_data_loader.get_all_data()
            with torch.no_grad():
                if type(self.model) == CVAEMARHMM:
                    preds_train = self.model.predict(x_train, sub_ids)
                    nll, _ = self.model.forward(x_train, sub_ids, epoch)
                else:
                    preds_train = self.model.predict(x_train)
                    nll = self.model.forward(x_train)
            
            # Handle MARHMM burn-in for train data
            if hasattr(self.model, 'max_lag') and self.model.max_lag > 0:
                preds_train = preds_train[:, self.model.max_lag:]
                y_train = y_train[:, self.model.max_lag:]
            
            y_train_np = y_train.detach().cpu().numpy().flatten()
            preds_train_np = preds_train.detach().cpu().numpy().flatten()
            train_nmi = calculate_nmi(preds_train_np, y_train_np)
            # Store with 'train_' prefix to distinguish from val NMI
            # Note: logger will add 'train/' prefix when logging
            self.validations[epoch]["train_nmi"] = train_nmi
            
            # Compute state entropy and perplexity for training data
            train_entropy_metrics = self.__compute_state_entropy(preds_train_np, self.train_data_loader.get_num_states())
            self.validations[epoch]["train_entropy"] = train_entropy_metrics["entropy"]
            self.validations[epoch]["train_perplexity"] = train_entropy_metrics["perplexity"] 
            self.validations[epoch]["train_nll"] = nll.item()
            self.validations[epoch]["train_log_likelihood"] = -nll.item()
            self.validations[epoch]["train_likelihood"] = np.exp(-nll.item())
            
            train_perp = train_entropy_metrics["perplexity"] / self.train_data_loader.get_num_states()
            train_ll = self.validations[epoch]["train_likelihood"]
            eps = 1e-12
            w = 0.9
            train_nlpp = w * np.log(train_ll + eps) + (1 - w) * np.log(train_perp + eps)
            self.validations[epoch]["train_nlpp"] = train_nlpp  
        
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
        self.model.prepare_for_training()
            
        # Save historic values for this epoch
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.historic_values:
                self.historic_values[name].append(p.detach().cpu().numpy())
        
        self.model.train()  # Set back to training mode
        if self.global_config.verbose:
            nmi_val = self.validations[epoch].get('nmi', 'N/A')
            train_nmi_val = self.validations[epoch].get('train_nmi', 'N/A')
            acc_val = self.validations[epoch].get('accuracy', 'N/A')
            perp_val = self.validations[epoch].get('perplexity', 'N/A')
            train_perp_val = self.validations[epoch].get('train_perplexity', 'N/A')
            train_nmi_str = f"{train_nmi_val:.4f}" if isinstance(train_nmi_val, (int, float)) else str(train_nmi_val)
            nmi_str = f"{nmi_val:.4f}" if isinstance(nmi_val, (int, float)) else str(nmi_val)
            acc_str = f"{acc_val:.4f}" if isinstance(acc_val, (int, float)) else str(acc_val)
            perp_str = f"{perp_val:.2f}" if isinstance(perp_val, (int, float)) else str(perp_val)
            train_perp_str = f"{train_perp_val:.2f}" if isinstance(train_perp_val, (int, float)) else str(train_perp_val)
            print(f"Epoch {epoch + 1} - Val NMI: {nmi_str}, Train NMI: {train_nmi_str}, "
                  f"Val Perp: {perp_str}/{self.data_loader.get_num_states()}, "
                  f"Train Perp: {train_perp_str}/{self.train_data_loader.get_num_states() if self.train_data_loader else 'N/A'}, "
                  f"Acc: {acc_str}")
            # print(f"Epoch {epoch + 1} - NMI: {nmi_str}, Accuracy: {acc_str}")
    
    
    
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
        summarize_metrics(validations, n_runs=len(train_details), attach_to=validations, namespace="summary", style="nested")
        with open(f"{self.global_config.results_dir}/{self.global_config.run_name}/validations.json", "w") as f:
            json.dump(validations, f)
        return validations

    def validate_data(self):
        x, y, _ = self.data_loader.get_all_data()
        has_features = self.data_loader.has_features_enabled()
        self.data_validations = {}
        out_path = f"{self.global_config.results_dir}/{self.global_config.run_name}/data_validations.json"
        if self.config.state_distinctness and has_features:
            distinctness = compute_state_distinctness(x, y)
            self.data_validations.update(distinctness)
        if self.config.summary_statistics:
            self.data_validations.update(compute_summary_statistics(x, y, self.data_loader))
            if has_features:
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
    
    def validate_cvae(self):
        assert type(self.model) == CVAEMARHMM, "Model must be of type CVAEMARHMM to validate CVAE latent representations."
        x, y, sub_ids = self.data_loader.get_all_data()
        self.model.prepare_for_inference()
        with torch.no_grad():
            x_latent = self.model.get_latent_representation(x, sub_ids)
        self.data_validations.update(compute_feature_statistics(x_latent, y, self.data_loader, True))
        distinctness = compute_state_distinctness(x_latent, y)
        self.data_validations.update(distinctness)
    
    def validate_cvae_epoch(self, epoch: int):
        self.validations[epoch] = {}
        assert type(self.model) == CVAEMARHMM, "Model must be of type CVAEMARHMM to validate CVAE latent representations."
        x, y, sub_ids = self.data_loader.get_all_data()
        self.model.prepare_for_inference()
        with torch.no_grad():
            x_latent_seq = self.model.get_latent_representation(x, sub_ids)  # (B,T,L)
            x_latent = x_latent_seq.reshape(-1, x_latent_seq.shape[-1])
        n_clusters = len(torch.unique(y))
        nmi = self.__calculate_kmeans_nmi(x_latent, y, n_clusters)
        self.validations[epoch]['cvae_latent_kmeans_nmi'] = nmi
        print(f"CVAE Latent KMeans NMI: {nmi:.4f}")

        # Trajectory metrics whenever sequence_length > 1; cheap and prior-agnostic.
        if x_latent_seq.dim() == 3 and x_latent_seq.shape[1] > 1:
            try:
                autocorr = latent_autocorr(x_latent_seq, lag=1)
                self.validations[epoch]['latent_autocorr_lag1'] = autocorr
                print(f"Latent autocorr (lag 1): {autocorr:.4f}")
            except Exception as e:
                print(f"Skipping latent autocorr: {e}")

        self._validate_cvae_prior_metrics(epoch, x, y, sub_ids)

        self.model.prepare_for_training()

    def _validate_cvae_prior_metrics(
        self,
        epoch: int,
        x: torch.Tensor,
        y: torch.Tensor,
        sub_ids: torch.Tensor,
    ) -> None:
        """Prior predictions, collapse metrics, and thesis checkpoint score S(e)."""
        k_pred = self.data_loader.get_num_states()
        prior_name = getattr(self.model.cvae, "prior", None)
        cs_cfg = self.global_config.trainer.checkpoint_score
        beta = cs_cfg.beta

        with torch.no_grad():
            val_nll, _ = self.model.forward(x, sub_ids, epoch)
        log_likelihood = float(-val_nll.item())
        self.validations[epoch]["log_likelihood"] = log_likelihood

        try:
            with torch.no_grad():
                if prior_name in ("hmm_gmm", "warm_hmm_gmm"):
                    y_hat, _, _ = self.model.cvae.predict_hmm_labels(x, sub_ids)
                else:
                    _, y_hat, _ = self.model.predict_gmm(
                        x, sub_ids, initialize_if_needed=True
                    )
        except Exception as e:
            print(f"Skipping CVAE prior metrics at epoch {epoch + 1}: {e}")
            return

        y_np = y.detach().cpu().numpy().flatten()
        preds_np = y_hat.detach().cpu().numpy().flatten()
        prior_nmi = float(calculate_nmi(preds_np, y_np))
        ent_norm = entropy_norm(preds_np, k_pred)
        score = checkpoint_score(log_likelihood, ent_norm, beta=beta)

        self.validations[epoch][PRIOR_PRED_NMI_KEY] = prior_nmi
        self.validations[epoch]["entropy_norm"] = ent_norm
        self.validations[epoch]["checkpoint_score"] = score
        self.validations[epoch]["n_unique_pred_states"] = int(len(np.unique(preds_np)))

        if y_hat.dim() == 2 and y_hat.shape[1] > 1:
            switch = state_switch_rate(y_hat)
            self.validations[epoch]["prior_switch_rate_per100"] = switch
        else:
            self.validations[epoch]["prior_switch_rate_per100"] = 0.0

        if self.global_config.verbose:
            sw = self.validations[epoch]["prior_switch_rate_per100"]
            print(
                f"Prior val — NMI: {prior_nmi:.4f}, log-lik: {log_likelihood:.4f}, "
                f"H_norm: {ent_norm:.4f}, S: {score:.4f}, n_states: "
                f"{self.validations[epoch]['n_unique_pred_states']}, switch/100: {sw:.4f}"
            )
    
    @staticmethod
    def _log_prior_label_stats(y_hat: torch.Tensor, y_true: torch.Tensor, prefix: str) -> None:
        yh = y_hat.detach().cpu().numpy().ravel()
        yt = y_true.detach().cpu().numpy().ravel()
        print(
            f"{prefix} labels: pred unique={np.unique(yh).tolist()} "
            f"(n={len(np.unique(yh))}), true unique={np.unique(yt).tolist()} "
            f"(n={len(np.unique(yt))})"
        )

    def validate_cvae_gmm(self):
        assert type(self.model) == CVAEMARHMM, "Model must be of type CVAEMARHMM to validate CVAE latent representations."
        x, y, sub_ids = self.data_loader.get_all_data()
        x_non_norm = self.data_loader.get_non_normalized_data()
        self.model.prepare_for_inference()
        with torch.no_grad():
            x_latent, y_hat, likelihood = self.model.predict_gmm(
                x, sub_ids, initialize_if_needed=True
            )
        self._log_prior_label_stats(y_hat, y, "GMM marginal")
        nmi = calculate_nmi(y_hat.detach().cpu().numpy().flatten(), y.detach().cpu().numpy().flatten())
        gmm_switch = state_switch_rate(y_hat) if y_hat.dim() == 2 and y_hat.shape[1] > 1 else 0.0
        print(f"GMM marginal switch rate (per 100): {gmm_switch:.4f}")
        return nmi, likelihood, y_hat, y, x_latent, x_non_norm, sub_ids, gmm_switch

    def validate_cvae_hmm(self):
        """Parallel of ``validate_cvae_gmm`` for the HMM-GMM prior (Viterbi labels)."""
        assert type(self.model) == CVAEMARHMM, "Model must be of type CVAEMARHMM."
        x, y, sub_ids = self.data_loader.get_all_data()
        x_non_norm = self.data_loader.get_non_normalized_data()
        self.model.prepare_for_inference()
        with torch.no_grad():
            y_hat, log_pz, mu = self.model.cvae.predict_hmm_labels(x, sub_ids)
            gmm_switch = 0.0
            try:
                _, y_gmm, _ = self.model.predict_gmm(x, sub_ids)
                self._log_prior_label_stats(y_gmm, y, "GMM marginal")
                gmm_switch = state_switch_rate(y_gmm)
            except Exception as e:
                print(f"Skipping GMM marginal comparison: {e}")
        self._log_prior_label_stats(y_hat, y, "HMM Viterbi")
        nmi = calculate_nmi(y_hat.detach().cpu().numpy().flatten(), y.detach().cpu().numpy().flatten())
        switch_rate = state_switch_rate(y_hat)
        print(f"HMM switch rate (per 100): {switch_rate:.4f}; GMM marginal (per 100): {gmm_switch:.4f}")
        return nmi, log_pz, y_hat, y, mu, x_non_norm, sub_ids, switch_rate, gmm_switch
    
    ####### HELPER METHODS #######
    
    def __calculate_kmeans_nmi(self, x_latent: torch.Tensor, labels: torch.Tensor, n_clusters: int):
        """Calculate KMeans clustering nmi on latent representations."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        try:
            x_np = x_latent.detach().cpu().numpy()
        except AttributeError:
            x_np = np.asarray(x_latent)
        if not np.isfinite(x_np).all():
            print("WARNING: NaN/Inf in CVAE latents; skipping KMeans NMI for this epoch.")
            return 0.0
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(x_np)
        # compute nmi
        nmi = calculate_nmi(cluster_labels, labels.detach().cpu().numpy().flatten())
        return nmi
        

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
    
    
    def __compute_state_entropy(self, predictions: np.ndarray, num_states: int) -> dict[str, float]:
        """Compute state usage entropy and perplexity (effective number of states).
        
        Args:
            predictions: Array of predicted state labels
            num_states: Total number of states in the model
            
        Returns:
            Dictionary with 'entropy' and 'perplexity' keys
        """
        # Count state frequencies
        counts = np.bincount(predictions.astype(int), minlength=num_states)
        total = counts.sum()
        
        if total == 0:
            return {"entropy": 0.0, "perplexity": 0.0}
        
        # Compute probability distribution
        probs = counts / total
        
        # Compute entropy: H = -sum(p * log(p))
        # Use base-e logarithm for natural entropy
        entropy = 0.0
        for p in probs:
            if p > 0:  # avoid log(0)
                entropy -= p * np.log(p)
        
        # Perplexity = exp(entropy) = effective number of states
        perplexity = np.exp(entropy)
        
        return {
            "entropy": float(entropy),
            "perplexity": float(perplexity)
        }
    
    
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
        return self.historic_values

    def get_data_validations(self) -> dict[str, Any]:
        assert self.data_validations, "Data validation has not been run yet."
        return self.data_validations