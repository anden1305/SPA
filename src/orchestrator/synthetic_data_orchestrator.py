
from pathlib import Path
from src.synthetic_data.synthetic_data_generator import build_generator, load_config

class SyntheticDataOrchestrator:
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
    
    def generate_data(self):
        cfg = load_config(self.config_path)
        gen = build_generator(cfg)
        eeg, labels = gen.generate()
        gen.save(eeg, labels, self.config_path)