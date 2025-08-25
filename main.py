



if __name__ == "__main__":
    import argparse
    from src.orchestrator.orchestrator import Orchestrator
    
    parser = argparse.ArgumentParser(description="Run SPA Orchestrator with a given config file")
    parser.add_argument("--config_path", "-c", required=True, help="Path to the configuration file")
    args = parser.parse_args()
    config_path = args.config_path
    
    orchestrator = Orchestrator(config_path)
    orchestrator.run()