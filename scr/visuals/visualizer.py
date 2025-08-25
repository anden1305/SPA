
from scr.config.config import GlobalConfig
from scr.data.data_loader import DataLoader
from scr.models.base_model import MLModel
from scr.training.trainer import Trainer
import matplotlib.pyplot as plt

class Visualizer:
    
    
    def __init__(self,
                 data_loader: DataLoader,
                 trainer: Trainer,
                 config: GlobalConfig):
        self.data_loader = data_loader
        self.trainer = trainer
        self.global_config = config
        self.config = self.global_config.visualizer
    
    def visualize(self):
        if self.config.losses:
            self.plot_losses()

    def plot_losses(self):
        losses = self.trainer.get_losses()
        plt.plot(losses)
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"{self.global_config.results_dir}/{self.global_config.run_name}/losses.png")
