
if __name__ == "__main__":
    import argparse
    from src.orchestrator.orchestrator import Orchestrator
    from src.orchestrator.synthetic_data_orchestrator import SyntheticDataOrchestrator
    from src.training.wandb_sweep_runner import run_sweep

    parser = argparse.ArgumentParser(description="Run SPA Orchestrator with a given config file")
    parser.add_argument("--method", "-m", default="train", help="Method to run (train | train_vae | generate | explore_synthetic | sweep)")
    parser.add_argument("--config_path", "-c", required=True, help="Path to the configuration file")
    parser.add_argument("--profile", "-p", action="store_true", help="Enable cProfile profiling for the run (train only)")
    args = parser.parse_args()
    method = args.method
    config_path = args.config_path

    if method not in ["train", "train_vae", "generate", "explore_synthetic", "sweep"]:
        raise ValueError(f"Unknown method: {method}")
    
    # Enforce that profiling is only valid for training runs
    if args.profile:
        assert method == "train", "--profile is only supported with --method train"

    if method == "train":
        orchestrator = Orchestrator(config_path, profile=args.profile)
        orchestrator.run()
    if method == "train_vae":
        orchestrator = Orchestrator(config_path, profile=args.profile)
        orchestrator.train_cvae()
    elif method == "generate":
        data_orchestrator = SyntheticDataOrchestrator(config_path)
        data_orchestrator.generate_data()
    elif method == "explore_synthetic":
        data_orchestrator = SyntheticDataOrchestrator(config_path)
        data_orchestrator.explore_synthetic()
    elif method == "sweep":
        run_sweep(config_path)