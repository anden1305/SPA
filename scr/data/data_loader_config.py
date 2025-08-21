from pydantic import BaseModel, Field, ValidationError, ConfigDict


class DataLoaderConfig(BaseModel):
    batch_size: int = Field(default=32, ge=1)
    transforms: list[str] = Field(default_factory=list)
    verbose: bool = Field(default=True)
    