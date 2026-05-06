"""Coordinator for sleep EEG noise analyses across labs."""

import subprocess
import sys
import time
from pathlib import Path

# Ensure module path
THIS_DIR = Path(__file__).parent
REPO_ROOT = THIS_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def run_analysis_script(script_name: str, description: str, output_dir: Path) -> int:
    """Run a noise analysis script using subprocess.
    
    Returns 0 on success, 1 on failure.
    """
    script_path = THIS_DIR / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return 1
    
    print(f"\n{'=' * 80}")
    print(f"🔊 {description}")
    print(f"{'=' * 80}")
    print(f"Running: {script_name}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path),
             "--out-dir", str(output_dir)],
            cwd=str(REPO_ROOT),
            capture_output=False,
            text=True,
            timeout=3600  # 1 hour timeout per script
        )
        if result.returncode != 0:
            print(f"❌ Script failed with exit code {result.returncode}")
            return 1
        print(f"✓ {script_name} completed successfully")
        return 0
    except subprocess.TimeoutExpired:
        print(f"❌ Script timed out: {script_name}")
        return 1
    except Exception as exc:
        print(f"❌ Error running script: {exc}")
        return 1


def run_analysis_script_with_args(script_name: str, description: str, output_dir: Path, extra_args: list[str]) -> int:
    """Run a noise analysis script with extra CLI arguments."""
    script_path = THIS_DIR / script_name
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return 1

    print(f"\n{'=' * 80}")
    print(f"🔊 {description}")
    print(f"{'=' * 80}")
    print(f"Running: {script_name}")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--out-dir", str(output_dir), *extra_args],
            cwd=str(REPO_ROOT),
            capture_output=False,
            text=True,
            timeout=3600,
        )
        if result.returncode != 0:
            print(f"❌ Script failed with exit code {result.returncode}")
            return 1
        print(f"✓ {script_name} completed successfully")
        return 0
    except subprocess.TimeoutExpired:
        print(f"❌ Script timed out: {script_name}")
        return 1
    except Exception as exc:
        print(f"❌ Error running script: {exc}")
        return 1


def main():
    print("=" * 80)
    print("SLEEP EEG NOISE ANALYSIS (Labs 2, 3, 5)")
    print("=" * 80)
    print(f"Analysis started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    output_dir = (REPO_ROOT / "results" / "noise").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyses = [
        ("emg_rms_awake_over_rem_lab2.py", "ANALYSIS 1/6: EMG RMS(Awake)/RMS(NREM) – Lab 2"),
        ("emg_rms_awake_over_rem_lab3.py", "ANALYSIS 2/6: EMG RMS(Awake)/RMS(NREM) – Lab 3"),
        ("emg_rms_awake_over_rem_lab5.py", "ANALYSIS 3/6: EMG RMS(Awake)/RMS(NREM) – Lab 5"),
        ("relative_power_lab2.py", "ANALYSIS 4/6: Relative Power (0.5–2 Hz)/(0.5–30 Hz) – Lab 2"),
        ("relative_power_lab3.py", "ANALYSIS 5/6: Relative Power (0.5–2 Hz)/(0.5–30 Hz) – Lab 3"),
        ("relative_power_lab5.py", "ANALYSIS 6/6: Relative Power (0.5–2 Hz)/(0.5–30 Hz) – Lab 5"),
    ]
    
    failed = []
    try:
        for i, (script_name, description) in enumerate(analyses, 1):
            ret = run_analysis_script(script_name, description, output_dir)
            if ret != 0:
                failed.append(script_name)
    except Exception as exc:
        print(f"❌ Unexpected error during analysis: {exc}")
        return 1
    
    # Summary
    print("\n" + "=" * 80)
    if not failed:
        print("🎉 ALL NOISE ANALYSES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results saved to: {output_dir}")
        print(f"Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return 0
    else:
        print("⚠️  ANALYSIS COMPLETED WITH ERRORS")
        print("=" * 80)
        print(f"Failed scripts: {', '.join(failed)}")
        print(f"Results (partial) saved to: {output_dir}")
        print(f"Analysis completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
