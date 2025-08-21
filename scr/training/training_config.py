from pydantic import BaseModel, Field, ValidationError, ConfigDict


class TrainingConfig(BaseModel):
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    optimizer: str = Field(default="adam", regex="^(adam|sgd|rmsprop)$")
    grad_clip: float | None = Field(default=None, ge=0, description="Gradient clipping value. If None, no clipping is applied.")
    verbose: bool = Field(default=False, description="Whether to print detailed training progress.")