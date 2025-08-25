



if __name__ == "__main__":
    
    from src.orchestrator.orchestrator import Orchestrator

    config_path = "src/config/run/config.yaml"

    orchestrator = Orchestrator(config_path)
    orchestrator.run()