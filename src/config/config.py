from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any
import yaml

class EarlyStoppingConfig(BaseModel):
    enabled: bool = Field(True, description="Enable adaptive early stopping.")
    patience: int = Field(..., ge=1, le=100, description="Epochs without relative improvement before action.")
    min_delta: float = Field(..., ge=0, le=0.5, description="Minimum relative improvement fraction required (e.g. 0.01 = 1%). Always relative.")

class TrainerConfig(BaseModel):
    epochs: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0)
    optimizer: str = Field(..., pattern="^(adam|sgd|rmsprop)$")
    grad_clip: float | None = Field(..., ge=0, description="Gradient clipping value. If None, no clipping is applied.")
    validate_per_epoch: int = Field(..., ge=0, le=50, description=("Frequency (in epochs) to run full validation & supervised metrics. "
            "Set to 0 to skip metric validation entirely; unsupervised early stopping "
            "(parameter-drift) still runs each epoch if early stopping is enabled. "
            "Maximum value is 50."))
    early_stopping: EarlyStoppingConfig | None = Field(default_factory=EarlyStoppingConfig, description="Early stopping settings. Set enabled=False to disable.")

class ValidatorConfig(BaseModel):
    nmi: bool = Field(..., description="Whether to compute NMI.")
    accuracy: bool = Field(..., description="Whether to compute accuracy.")
    cross_nmi: bool = Field(..., description="Whether to compute cross NMI.")
    state_distinctness: bool = Field(..., description="Whether to compute state distinctness.")
    summary_statistics: bool = Field(..., description="Whether to compute summary statistics.")
class VisualizerConfig(BaseModel):
    losses: bool = Field(..., description="Whether to visualize losses.")
    pca_tripanel: bool = Field(..., description="Whether to visualize PCA tripanel.")
    confusion_matrix: bool = Field(..., description="Whether to visualize confusion matrix.")
    state_distinctness: bool = Field(..., description="Whether to visualize state distinctness.")
    summary_statistics: bool = Field(..., description="Whether to visualize summary statistics.")
    historic_values: bool = Field(..., description="Whether to visualize historic model values (parameters & confusion matrices).")

class TransformsConfig(BaseModel):
    type: str = Field(..., description="Type of transform, e.g., 'fft'.")
    params: dict[str, Any] = Field(...)

class DataLoaderConfig(BaseModel):
    batch_size: int | None = Field(..., ge=1, description="Number of windows. If None, use full dataset.")
    transforms: list[TransformsConfig] = Field(default_factory=list)
    shuffle: bool = Field(..., description="Whether to shuffle data each epoch.")
    normalize: bool = Field(..., description="Whether to normalize data using mean and std.")

class DatasetConfig(BaseModel):
    type: str = Field(..., pattern="^(synthetic|mssv)$")
    id: str = Field(..., description="Dataset identifier or path.")
    run: int | None = Field(default=None, ge=1)

class ModelConfig(BaseModel):
    type: str = Field(..., pattern="^(hmm|marhmm)$", description="Type of model, e.g., 'hmm'.")
    init_strategy: str = Field(..., pattern="^(random_uniform|random_dirichlet|random_separated|kmeans|kmeans_pca)$", description="Initialization strategy for the model.")
    init_noisy: bool = Field(...)
    params: dict[str, Any] = Field(..., description="Model-specific parameters.")

class GlobalConfig(BaseModel):
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    visualizer: VisualizerConfig = Field(default_factory=VisualizerConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    verbose: bool = Field(..., description="Whether to print detailed logs.")
    seed: int = Field(..., ge=0, description="Random seed for reproducibility.")
    results_dir: str = Field(default="results/training", description="Directory to save results.")
    run_name: str = Field(..., description="Name of the current run.")
    runs: int = Field(..., ge=1, description="Number of runs to execute.")
    validate_data: bool = Field(..., description="Whether to perform data validation before training.")

    @classmethod
    def from_yaml(cls, file_path: str) -> "GlobalConfig":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config YAML not found: {path}")
        with path.open("r") as f:
            data: Any = yaml.safe_load(f) or {}
        return cls.model_validate(data)