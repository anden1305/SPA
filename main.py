



if __name__ == "__main__":
    import argparse
    from src.orchestrator.orchestrator import Orchestrator
    from src.orchestrator.synthetic_data_orchestrator import SyntheticDataOrchestrator

    parser = argparse.ArgumentParser(description="Run SPA Orchestrator with a given config file")
    parser.add_argument("--method", "-m", default="train", help="Method to run (train or generate)")
    parser.add_argument("--config_path", "-c", required=True, help="Path to the configuration file")
    args = parser.parse_args()
    method = args.method
    config_path = args.config_path

    if method not in ["train", "generate", "explore_synthetic"]:
        raise ValueError(f"Unknown method: {method}")
    
    if method == "train":
        orchestrator = Orchestrator(config_path)
        orchestrator.run()
    elif method == "generate":
        data_orchestrator = SyntheticDataOrchestrator(config_path)
        data_orchestrator.generate_data()
    elif method == "explore_synthetic":
        data_orchestrator = SyntheticDataOrchestrator(config_path)
        data_orchestrator.explore_synthetic()