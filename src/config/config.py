from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any
import yaml

class SchedulerConfig(BaseModel):
    enabled: bool = Field(..., description="Whether to use a learning rate scheduler.")
    type: str = Field(..., pattern="^(step|exponential)$", description="Type of scheduler, e.g., 'step'.")
    step_size: int | None = Field(..., description="Step size for 'step' scheduler.")
    gamma: float = Field(..., gt=0, lt=1, description="Decay factor for the scheduler.")

class EarlyStoppingConfig(BaseModel):
    enabled: bool = Field(True, description="Enable adaptive early stopping.")
    patience: int = Field(..., ge=1, le=100, description="Epochs without relative improvement before action.")
    min_delta: float = Field(..., ge=0, le=0.5, description="Minimum relative improvement fraction required (e.g. 0.01 = 1%). Always relative.")
    # Hardcoded in src/training/early_stopping.py:
    #   - _MIN_LR = 1e-5 (minimum learning rate floor)
    #   - _LR_FACTOR = 0.5 (learning rate reduction factor)
    #   - _EMA_ALPHA = 0.3 (exponential moving average decay)
    #   - _WARMUP_VALIDATIONS = 5 (number of warmup validations)
    #   - 1.01 multiplier for LR comparison threshold

class TrainerConfig(BaseModel):
    epochs: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0)
    optimizer: str = Field(..., pattern="^(adam|sgd|rmsprop|adamw)$")
    grad_clip: float | None = Field(..., ge=0, description="Gradient clipping value. If None, no clipping is applied.")
    validate_per_epoch: int = Field(..., ge=0, le=50, description="Frequency of validation during training in epochs. Set to 0 to disable per-epoch validation. Maximum value is 50.")
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig, description="Early stopping settings. Set enabled=False to disable.")
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

class ValidatorConfig(BaseModel):
    nmi: bool = Field(..., description="Whether to compute NMI.")
    accuracy: bool = Field(..., description="Whether to compute accuracy.")
    cross_nmi: bool = Field(..., description="Whether to compute cross NMI.")
    learning_rate: bool = Field(..., description="Whether to compute learning rate.")
    state_distinctness: bool = Field(..., description="Whether to compute state distinctness.")
    summary_statistics: bool = Field(..., description="Whether to compute summary statistics.")
    # No hardcoded constants - all validation metrics are configurable toggles

class VisualizerConfig(BaseModel):
    losses: bool = Field(..., description="Whether to visualize losses.")
    learning_rate: bool = Field(..., description="Whether to visualize learning rate.")
    pca_tripanel: bool = Field(..., description="Whether to visualize PCA tripanel.")
    confusion_matrix: bool = Field(..., description="Whether to visualize confusion matrix.")
    state_distinctness: bool = Field(..., description="Whether to visualize state distinctness.")
    summary_statistics: bool = Field(..., description="Whether to visualize summary statistics.")
    historic_values: bool = Field(..., description="Whether to visualize historic model values (parameters & confusion matrices).")
    # Hardcoded in src/visuals/visualizer.py (aesthetic constants)

class TransformsConfig(BaseModel):
    type: str = Field(..., description="Type of transform, e.g., 'fft'.")
    channel: str = Field(..., description="Channel group the transform applies to, e.g., 'EEG'.")
    params: dict[str, Any] = Field(...)

class DataLoaderConfig(BaseModel):
    batch_size: int | None = Field(..., ge=1, description="Number of windows. If None, use full dataset.")
    window_size: int | None = Field(..., ge=1, description="Size of each data window in timesteps.")
    stride: int | None = Field(..., ge=1, description="Stride between windows in timesteps.")
    transforms: list[TransformsConfig] = Field(default_factory=list)
    shuffle: bool = Field(..., description="Whether to shuffle data each epoch.")
    normalize: bool = Field(..., description="Whether to normalize data using mean and std.")
    # No hardcoded constants - all data loading behavior is configurable

class DatasetConfig(BaseModel):
    type: str = Field(..., pattern="^(synthetic|mssv)$")
    id: str = Field(..., description="Dataset identifier or path.")
    run: int | None = Field(default=None, ge=1)
    # No hardcoded constants - dataset selection is fully configurable

class ModelConfig(BaseModel):
    type: str = Field(..., pattern="^(hmm|marhmm)$", description="Type of model, e.g., 'hmm'.")
    covariance_type: str = Field(..., pattern="^(diag|full|meanonly)$", description="Covariance structure: 'diag' (diagonal), 'full' (Cholesky-factorized with softplus), or 'meanonly' (identity, HMM only).")
    init_strategy: str = Field(..., pattern="^(random_uniform|random_dirichlet|random_separated|kmeans|kmeans_pca)$", description="Initialization strategy for the model.")
    init_noisy: bool = Field(...)  # Adds Gaussian noise to initialization; noise levels hardcoded below
    params: dict[str, Any] = Field(..., description="Model-specific parameters.")
    # MAR-HMM params (via params dict - these ARE configurable):
    #   - ridge: L2 penalty on regression coefficients (default 0.0)
    #   - var_reg: variance stabilization penalty (default 0.0)
    #   - sticky_coef: sticky transition prior weight (default 0.0)
    #   - sticky_kappa: self-transition probability in prior (default 0.9)
    #   - max_lag: autoregressive order (required for marhmm)
    #
    # Hardcoded in src/models/hmm.py & marhmm.py:
    #   - self.jitter = 1e-5 (numerical stability for covariance matrices)
    #   - Gaussian log-likelihood constants: -0.5 factor, log(2π)
    #
    # Hardcoded in src/models/hmm.py (commented out regularization):
    #   - max_var_threshold = 10.0
    #   - min_variance_threshold = 1
    #
    # Hardcoded in src/models/marhmm.py (regularization):
    #   - min_var = 1e-3 (variance floor for diag covariance)
    #   - min_diag = 1e-3 (diagonal floor for full covariance)
    #   - 1e-3 penalty weight for variance constraints
    #   - 1e-4 variance threshold for penalty
    #   - 0.0 zero-likelihood for early timesteps (lp[:, :max_lag, :] = 0.0)
    #
    # Hardcoded in src/initializations/kmeans.py:
    #   - 1e-6 minimum variance clamp
    #   - 1e-4 Cholesky stabilization jitter
    #   - 1e-8 minimum diagonal clamp
    #   - 1e-3 smoothing for pi_counts and trans_counts
    #   - 1.0/S uniform initialization fallback
    #   - alpha=1.0 for Dirichlet initialization
    #
    # Hardcoded in src/models/hmm.py add_noise() method (when init_noisy=True):
    #   - mean_std: 0.05
    #   - cov_noise_std: 0.05
    #   - init_logits_std: 0.05
    #   - self_transition_bias: 0.05
    #   - spread: 0.05
    #   - jitter_std: 0.05
    #
    # Hardcoded in src/models/marhmm.py __initialize_weights():
    #   - coeff_std: 0.03
    #   - jitter_std: 0.05
    #   - var_init: 1.0

class GlobalConfig(BaseModel):
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    visualizer: VisualizerConfig = Field(default_factory=VisualizerConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    train_datasets: list[DatasetConfig] = Field(default_factory=list)
    val_datasets: list[DatasetConfig] = Field(default_factory=list)
    model: ModelConfig = Field(default_factory=ModelConfig)
    verbose: bool = Field(..., description="Whether to print detailed logs.")
    seed: int = Field(..., ge=0, description="Random seed for reproducibility.")
    results_dir: str = Field(default="results/training", description="Directory to save results.")
    run_name: str = Field(..., description="Name of the current run.")
    runs: int = Field(..., ge=1, description="Number of runs to execute.")
    validate_data: bool = Field(..., description="Whether to perform data validation before training.")
    # No hardcoded constants at global level - all experiment settings are configurable

    @classmethod
    def from_yaml(cls, file_path: str) -> "GlobalConfig":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config YAML not found: {path}")
        with path.open("r") as f:
            data: Any = yaml.safe_load(f) or {}
        return cls.model_validate(data)