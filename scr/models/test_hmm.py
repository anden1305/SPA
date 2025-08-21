

from torch import Tensor
from scr.config.config import GlobalConfig
from scr.models.base_model import MLModel


class TestHMM(MLModel):
    def __init__(self, config: GlobalConfig):
        super().__init__(config)
        self.num_states = self.dataset.states
        # self.num_states = int(num_states)
        # self.obs_dim = int(obs_dim)
        # if device is None:
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device(device)
        # self.normalize_time = normalize_time  # if True, forward returns average log-prob per time step
        # self.covariance_type = covariance_type.lower()
        # if self.covariance_type not in {"diag", "full", "meanonly"}:
        #     raise ValueError("covariance_type must be one of {'diag','full','meanonly'}")
        # self.jitter = float(jitter)
        
    def forward(self, x: Tensor) -> Tensor:
        return x
    
    def __str__(self) -> str:
        return "NewHMM()"