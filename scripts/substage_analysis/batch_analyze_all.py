"""Batch process all HMM and MARHMM results for substage analysis.

This script finds all result directories in the substages_analysis folders and runs:
1. export_predictions_to_npz.py (if results.npz doesn't exist or --force flag is set)
2. run_substage_analysis.py (on the exported NPZ file)

Usage:
    python scripts/substage_analysis/batch_analyze_all.py
    python scripts/substage_analysis/batch_analyze_all.py --force  # Re-export even if NPZ exists
    python scripts/substage_analysis/batch_analyze_all.py --export-only  # Only export, no analysis
    python scripts/substage_analysis/batch_analyze_all.py --analyze-only  # Only analyze existing NPZ files
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def find_result_directories(base_dir: Path):
    """Find all result directories in a folder (dirs with config.json and 1/ subfolder)."""
    result_dirs = []
    if not base_dir.exists():
        return result_dirs
    
    for item in base_dir.iterdir():
        if item.is_dir():
            # Check if it has config.json and a run directory
            if (item / "config.json").exists():
                # Check for run directory (1/ or 0/)
                has_run_dir = (item / "1").exists() or (item / "0").exists()
                if has_run_dir:
                    result_dirs.append(item)
    
    return sorted(result_dirs)


def run_export(result_dir: Path, force: bool = False):
    """Run export_predictions_to_npz.py on a result directory."""
    npz_path = result_dir / "results.npz"
    
    if npz_path.exists() and not force:
        print(f"  ⏭️  Skipping export (results.npz already exists)")
        return True
    
    print(f"  🔄 Exporting predictions to NPZ...")
    
    try:
        cmd = [
            "uv", "run", "python",
            str(repo_root / "scripts/substage_analysis/export_predictions_to_npz.py"),
            str(result_dir),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        print(f"  ✅ Export complete")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Export failed:")
        print(f"     {e.stderr}")
        return False


def run_analysis(npz_path: Path):
    """Run run_substage_analysis.py on an NPZ file."""
    if not npz_path.exists():
        print(f"  ⚠️  NPZ file not found: {npz_path}")
        return False
    
    print(f"  🔄 Running substage analysis...")
    
    try:
        cmd = [
            "uv", "run", "python",
            str(repo_root / "scripts/substage_analysis/run_substage_analysis.py"),
            str(npz_path),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        print(f"  ✅ Analysis complete")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Analysis failed:")
        print(f"     {e.stderr}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all HMM and MARHMM results for substage analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-export NPZ files even if they already exist",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only export to NPZ, skip analysis",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only run analysis on existing NPZ files, skip export",
    )
    
    args = parser.parse_args()
    
    # Find all result directories
    base_results_dir = repo_root / "results" / "substages_analysis"
    hmm_dir = base_results_dir / "hmm"
    marhmm_dir = base_results_dir / "marhmm"
    
    print(f"\n{'='*80}")
    print(f"BATCH SUBSTAGE ANALYSIS")
    print(f"{'='*80}\n")
    
    hmm_results = find_result_directories(hmm_dir)
    marhmm_results = find_result_directories(marhmm_dir)
    
    all_results = [
        ("HMM", d) for d in hmm_results
    ] + [
        ("MARHMM", d) for d in marhmm_results
    ]
    
    print(f"Found {len(all_results)} result directories:")
    print(f"  • {len(hmm_results)} HMM runs")
    print(f"  • {len(marhmm_results)} MARHMM runs")
    print()
    
    if not all_results:
        print("No result directories found!")
        return
    
    # Process each directory
    export_success = 0
    export_skipped = 0
    export_failed = 0
    analysis_success = 0
    analysis_failed = 0
    
    for i, (model_type, result_dir) in enumerate(all_results, 1):
        print(f"\n[{i}/{len(all_results)}] Processing: {result_dir.name}")
        print(f"    Model: {model_type}")
        
        npz_path = result_dir / "results.npz"
        
        # Export step
        if not args.analyze_only:
            if run_export(result_dir, force=args.force):
                if npz_path.exists():
                    export_success += 1
                else:
                    export_skipped += 1
            else:
                export_failed += 1
                continue  # Skip analysis if export failed
        
        # Analysis step
        if not args.export_only:
            if run_analysis(npz_path):
                analysis_success += 1
            else:
                analysis_failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    
    if not args.analyze_only:
        print(f"\nExport:")
        print(f"  ✅ Success: {export_success}")
        print(f"  ⏭️  Skipped: {export_skipped}")
        print(f"  ❌ Failed:  {export_failed}")
    
    if not args.export_only:
        print(f"\nAnalysis:")
        print(f"  ✅ Success: {analysis_success}")
        print(f"  ❌ Failed:  {analysis_failed}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
