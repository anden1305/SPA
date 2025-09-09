

def test_pipeline():
    
    def print_completion_message():
        print("\n" + "="*50)
        print("✅  All pipeline tests have completed successfully!  ✅")
        print("="*50 + "\n")

    def print_failure_message(error):
        print("\n" + "="*50)
        print("❌  Pipeline test failed with error:  ❌")
        print(error)
        print("="*50 + "\n")
    
    def print_start_message():
        print("\n" + "="*50)
        print("🚀  Starting pipeline tests...  🚀")
        print("="*50 + "\n")
    
    import os
    import glob
    from src.orchestrator.orchestrator import Orchestrator
    config_files = glob.glob(os.path.join('tests/configs', '*.yaml'))
    print_start_message()
    try:
        for config_file in config_files:
            orchestrator = Orchestrator(config_file)
            orchestrator.run()
    except Exception as e:
        print_failure_message(e)
        raise Exception("Pipeline test failed with the above error.") from e
    else:
        print_completion_message()