



if __name__ == "__main__":
    
    from scr.orchestrator.orchestrator import Orchestrator

    config_path = "scr/config/config_no_features.yaml"

    orchestrator = Orchestrator(config_path)
    orchestrator.run()