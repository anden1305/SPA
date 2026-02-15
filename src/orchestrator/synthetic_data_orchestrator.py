
from pathlib import Path
from scripts.synthetic_data_exploration.synthetic_data_exploration import BASE_RESULTS, epoch_std_distribution, load_folder, mean_power_spectrum, raw_eeg_excerpt, summarize_labels
from src.synthetic_data.synthetic_data_generator import build_generator, load_config

class SyntheticDataOrchestrator:
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
    
    def generate_data(self):
        cfg = load_config(self.config_path)
        gen = build_generator(cfg)
        eeg, labels = gen.generate()
        gen.save(eeg, labels, self.config_path)
    
    def explore_synthetic(self):
        folder = self.config_path
        if not folder.exists():
            print(f"Provided folder does not exist: {folder}")
            return 1
        eeg, labels, meta = load_folder(folder)
        # Derive run name from folder (e.g., data/synthetic_data/demo_basic -> demo_basic)
        run_name = folder.name
        results = BASE_RESULTS / run_name
        results.mkdir(parents=True, exist_ok=True)
        # Save a small meta summary
        summary_path = results / "run_info.txt"
        summary_lines = [
            f"source_folder: {folder}",
            f"run_name: {run_name}",
            f"sampling_rate_hz: {meta.get('sampling_rate_hz','?')}",
            f"n_epochs: {len(labels)}",
            f"epoch_length_s: {meta.get('epoch_length_s', meta.get('config_used',{}).get('epoch_length_s','?'))}",
        ]
        summary_path.write_text("\n".join(summary_lines))
        summarize_labels(labels, meta, results)
        epoch_std_distribution(eeg, results)
        raw_eeg_excerpt(eeg, labels, meta, results, seconds=40, num_excerpts=10)
        mean_power_spectrum(eeg, labels, meta, results)
        print("Synthetic exploration v2 complete ->", results)
        return 0