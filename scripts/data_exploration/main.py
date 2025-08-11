"""
Main script for comprehensive data exploration of sleep EEG data.
This script orchestrates all data exploration modules and generates
a complete analysis of the dataset.
"""

import sys
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import exploration modules
from stage_differences import analyze_stage_differences

def main():
    """
    Main function to run complete data exploration.
    """
    print("=" * 80)
    print("COMPREHENSIVE SLEEP EEG DATA EXPLORATION")
    print("=" * 80)
    print(f"Analysis started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Define output directory
    output_dir = "results/data_exploration_2"
    
    try:
        # 1. Analyze signal differences between sleep stages
        print("📊 Starting Analysis 1: Signal Differences Between Sleep Stages")
        analyze_stage_differences(output_dir)
        print("✅ Analysis 1 completed successfully!\n")
        
        # TODO: Add other analysis modules as they are implemented
        # 2. analyze_participant_differences(output_dir)
        # 3. analyze_lab_differences(output_dir)
        # 4. analyze_eeg_placement_differences(output_dir)
        # 5. analyze_50hz_interference(output_dir)
        # 6. analyze_stage_transitions(output_dir)
        # 7. analyze_stage_durations(output_dir)
        
        print("=" * 80)
        print("🎉 ALL DATA EXPLORATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Analysis terminated.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
