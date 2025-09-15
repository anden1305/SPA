from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any
import yaml

class TrainerConfig(BaseModel):
    epochs: int = Field(..., ge=1)
    learning_rate: float = Field(..., gt=0)
    optimizer: str = Field(..., pattern="^(adam|sgd|rmsprop)$")
    grad_clip: float | None = Field(default=None, ge=0, description="Gradient clipping value. If None, no clipping is applied.")

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
    param_histories: bool = Field(..., description="Whether to visualize parameter histories.")

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