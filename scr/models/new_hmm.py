

from torch import Tensor
from scr.config.config import GlobalConfig
from scr.models.base_model import MLModel


class NewHMM(MLModel):
    def __init__(self, config: GlobalConfig):
        super().__init__(config)
        
    def forward(self, x: Tensor) -> Tensor:
        return x
    
    def __str__(self) -> str:
        return "NewHMM()"