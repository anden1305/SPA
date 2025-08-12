"""Coordinator for sleep EEG data exploration analyses."""

import sys
import time
from pathlib import Path

# Ensure module path
THIS_DIR = Path(__file__).parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from eeg_placement_differences import analyze_eeg_placement_differences
from interference_50hz import analyze_50hz_interference
from stage_transitions import analyze_stage_transitions
from stage_durations import analyze_stage_durations
from artifact_characteristics import analyze_artifact_characteristics
from lab_differences import analyze_lab_differences
from stage_differences import analyze_stage_differences
from participant_differences import analyze_participant_differences
from dataset_overview import analyze_dataset_overview


def main():
    print("=" * 80)
    print("COMPREHENSIVE SLEEP EEG DATA EXPLORATION")
    print("=" * 80)
    print(f"Analysis started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    output_dir = "results/data_exploration"

    try:
        # Overview (new)
        print("\n" + "=" * 80)
        print("📊 ANALYSIS 1/9: DATASET OVERVIEW")
        print("=" * 80)
        analyze_dataset_overview(output_dir)

        # Example disabled analyses
        analyze_stage_differences(output_dir)
        analyze_participant_differences(output_dir)
        analyze_eeg_placement_differences(output_dir)
        analyze_50hz_interference(output_dir=output_dir, exclude_lab1=False)

        print("\n" + "=" * 80)
        print("📊 ANALYSIS 5/9: LAB DIFFERENCES (incl. PSD)")
        print("=" * 80)
        # Include lab_1 only for this analysis
        analyze_lab_differences(output_dir=output_dir, participants_per_lab=10, exclude_lab1=False)

        print("\n" + "=" * 80)
        print("📊 ANALYSIS 6/9: STAGE TRANSITIONS")
        print("=" * 80)
        analyze_stage_transitions(output_dir=output_dir, exclude_lab1=True)

        print("\n" + "=" * 80)
        print("📊 ANALYSIS 7/9: STAGE DURATIONS")
        print("=" * 80)
        analyze_stage_durations(output_dir=output_dir, exclude_lab1=True)

        print("\n" + "=" * 80)
        print("📊 ANALYSIS 8/9: ARTIFACT CHARACTERISTICS")
        analyze_artifact_characteristics(output_dir=output_dir, exclude_lab1=True)

        print("\n" + "=" * 80)
        print("📊 ANALYSIS 9/9: (RESERVED / FUTURE)")
        print("=" * 80)
        # Placeholder for future analysis (kept for numbering consistency)
    
    except Exception as exc:
        print(f"❌ Error during analysis: {exc}")
        print("Analysis terminated.")
        return 1

    print("=" * 80)
    print("🎉 ALL DATA EXPLORATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
