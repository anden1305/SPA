from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any
import yaml

class TrainerConfig(BaseModel):
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    optimizer: str = Field(default="adam", pattern="^(adam|sgd|rmsprop)$")
    grad_clip: float | None = Field(default=None, ge=0, description="Gradient clipping value. If None, no clipping is applied.")

class ValidatorConfig(BaseModel):
    nmi: bool = Field(default=True)
    accuracy: bool = Field(default=True)
    cross_nmi: bool = Field(default=True)
    state_distinctness: bool = Field(default=True)
    summary_statistics: bool = Field(default=True)

class VisualizerConfig(BaseModel):
    losses: bool = Field(default=True)
    pca_tripanel: bool = Field(default=True)
    confusion_matrix: bool = Field(default=True)
    state_distinctness: bool = Field(default=True)
    summary_statistics: bool = Field(default=True)

class TransformsConfig(BaseModel):
    type: str = Field(default="default_transform")
    params: dict[str, Any] = Field(default_factory=dict)

class DataLoaderConfig(BaseModel):
    batch_size: int = Field(default=32, ge=1)
    transforms: list[TransformsConfig] = Field(default_factory=list)
    shuffle: bool = Field(default=True)
    normalize: bool = Field(default=True)

class DatasetConfig(BaseModel):
    type: str = Field(default="synthetic", pattern="^(synthetic|mssv)$")
    id: str = Field(default="default_dataset")
    normalize: bool = Field(default=True)
    run: int | None = Field(default=None, ge=1)

class ModelConfig(BaseModel):
    type: str = Field(default="hmm", pattern="^(hmm|new_hmm)$")
    init_strategy: str = Field(default="random", pattern="^(random_uniform|random_dirichlet|random_separated|kmeans|kmeans_pca)$")

class GlobalConfig(BaseModel):
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    validator: ValidatorConfig = Field(default_factory=ValidatorConfig)
    visualizer: VisualizerConfig = Field(default_factory=VisualizerConfig)
    dataloader: DataLoaderConfig = Field(default_factory=DataLoaderConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    verbose: bool = Field(default=False)
    seed: int = Field(default=42, ge=0)
    results_dir: str = Field(default="results/training")
    run_name: str = Field(default="default_run")
    runs: int = Field(default=1, ge=1)
    validate_data: bool = Field(default=True)

    @classmethod
    def from_yaml(cls, file_path: str) -> "GlobalConfig":
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Config YAML not found: {path}")
        with path.open("r") as f:
            data: Any = yaml.safe_load(f) or {}
        return cls.model_validate(data)