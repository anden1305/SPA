"""Generate missing validation performance plots for generalization experiments.

This script scans generalization results directories and creates validation_performance.png
plots for runs that completed but are missing the plot.

Usage:
    python scripts/generate_generalization_plots.py <generalization_results_dir>
    
Example:
    python scripts/generate_generalization_plots.py results/generalization/
    uv run scripts/generate_generalization_plots.py results/generalization/ --comparison
"""

import argparse
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns

# Set seaborn style for nicer plots
sns.set_theme(style="whitegrid", palette="husl")
sns.set_context("notebook", font_scale=1.1)

# Add repo root to path
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def find_generalization_runs(gen_dir: Path) -> list[Path]:
    """Find all completed generalization run directories.
    
    Args:
        gen_dir: Root generalization directory
        
    Returns:
        List of run directory paths (e.g., .../1/)
    """
    run_dirs = []
    
    if not gen_dir.exists():
        return run_dirs
    
    # Scan model type subdirectories (hmm, marhmm)
    for model_dir in gen_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        # Scan experiment directories (with timestamps)
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Look for run subdirectories (0, 1, 2, etc.)
            for run_dir in exp_dir.iterdir():
                if not run_dir.is_dir() or not run_dir.name.isdigit():
                    continue
                
                # Check if this is a valid run (has validations.json)
                if (run_dir / "validations.json").exists():
                    run_dirs.append(run_dir)
    
    return run_dirs


def generate_validation_plot(run_dir: Path) -> tuple[bool, Path]:
    """Generate validation performance plot for a single run (matching Visualizer style).
    
    Args:
        run_dir: Path to run directory
        
    Returns:
        Tuple of (success, output_path)
    """
    plots_dir = run_dir / "plots"
    output_file = plots_dir / "validation_performance.png"
    
    # Force regeneration - always create new plot
    
    # Load validations
    validations_path = run_dir / "validations.json"
    if not validations_path.exists():
        print(f"⚠️  No validations.json in {run_dir}")
        return False, None
    
    with open(validations_path, 'r') as f:
        validations = json.load(f)
    
    if not validations:
        print(f"⚠️  Empty validations in {run_dir}")
        return False, None
    
    # Get the latest validation epoch with log_likelihood data
    # The final epoch may not have LL, so search backwards
    latest_epoch_with_ll = None
    latest_val_with_ll = None
    for epoch in sorted([int(k) for k in validations.keys()], reverse=True):
        val = validations[str(epoch)]
        if val.get('log_likelihood') is not None:
            latest_epoch_with_ll = epoch
            latest_val_with_ll = val
            break
    
    # Get the absolute latest epoch for NMI (might be different)
    latest_epoch = max(int(k) for k in validations.keys())
    latest_val = validations[str(latest_epoch)]
    
    # Extract metrics - use LL from latest epoch that has it, NMI from absolute latest
    log_likelihood = latest_val_with_ll.get('log_likelihood') if latest_val_with_ll else None
    nmi = latest_val.get('nmi')
    
    # Skip if both metrics are missing
    if log_likelihood is None and nmi is None:
        print(f"⚠️  No metrics available in {run_dir}")
        return False, None
    
    # Extract test subject from directory name
    exp_name = run_dir.parent.name
    test_subject = "validation"
    if "sub" in exp_name:
        # Extract sub-XXX from name (without timestamp)
        parts = exp_name.split("_")
        for part in parts:
            if part.startswith("sub"):
                # Remove any bracketed timestamp like [20251217-231950]
                test_subject = part.split()[0]  # Take only first token before any space/bracket
                break
    
    # Create plots directory if needed
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plot (matching Visualizer.__plot_validation_performance_single)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    colors = sns.color_palette("husl", 2)
    
    # Log Likelihood
    if log_likelihood is not None:
        ax1.bar([0], [log_likelihood], color=colors[0], edgecolor='black', linewidth=1.5, alpha=0.85, width=0.5)
        ax1.set_ylabel('Validation Log Likelihood', fontsize=11)
        ax1.set_title('Predictive Log Likelihood', fontsize=12, fontweight='bold')
        ax1.set_xticks([0])
        ax1.set_xticklabels([f'{test_subject}'])
        ax1.grid(axis='y', alpha=0.3)
        
        # Better y-axis scaling to show the value clearly
        # Set limits to give 10% padding above and below
        if log_likelihood < 0:
            y_range = abs(log_likelihood) * 0.1
            ax1.set_ylim([log_likelihood - y_range, y_range])
        else:
            y_range = abs(log_likelihood) * 0.1
            ax1.set_ylim([-y_range, log_likelihood + y_range])
        
        # Position text in the middle of the bar for better visibility
        ax1.text(0, log_likelihood/2, 
                f'{log_likelihood:.4f}', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
    else:
        ax1.text(0.5, 0.5, 'No LL Data', ha='center', va='center', fontsize=14, 
                transform=ax1.transAxes, color='gray')
        ax1.set_title('Predictive Log Likelihood', fontsize=12, fontweight='bold')
        ax1.set_xticks([])
    
    # NMI
    if nmi is not None:
        ax2.bar([0], [nmi], color=colors[1], edgecolor='black', linewidth=1.5, alpha=0.85, width=0.5)
        ax2.set_ylabel('NMI vs True States', fontsize=11)
        ax2.set_title('NMI Performance', fontsize=12, fontweight='bold')
        ax2.set_xticks([0])
        ax2.set_xticklabels([f'{test_subject}'])
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        ax2.text(0, nmi + 0.02, 
                f'{nmi:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No NMI Data', ha='center', va='center', fontsize=14, 
                transform=ax2.transAxes, color='gray')
        ax2.set_title('NMI Performance', fontsize=12, fontweight='bold')
        ax2.set_xticks([])
        ax2.set_ylim([0, 1])
    
    # Create title showing which epochs were used
    if latest_epoch_with_ll and latest_epoch_with_ll != latest_epoch:
        title = f'Validation Performance on {test_subject} (NMI@{latest_epoch}, LL@{latest_epoch_with_ll})'
    else:
        title = f'Validation Performance on {test_subject} (Epoch {latest_epoch})'
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    return True, output_file


def generate_epoch_progression_plots(gen_dir: Path, model_type: str = None) -> list[Path]:
    """Generate plots showing validation metrics progression over epochs.
    
    Args:
        gen_dir: Root generalization directory
        model_type: Specific model type (hmm/marhmm) or None for all
        
    Returns:
        List of generated plot paths
    """
    # Collect epoch data by model type and subject
    data_by_model = defaultdict(dict)  # {model: {subject: {'epochs': [], 'll': [], 'nmi': []}}}
    generated_plots = []
    
    if not gen_dir.exists():
        return generated_plots
    
    # Scan model type subdirectories
    for model_dir in gen_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        current_model = model_dir.name
        if model_type and current_model != model_type:
            continue
        
        # Scan experiment directories
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Extract subject from directory name
            exp_name = exp_dir.name
            subject = None
            if "sub" in exp_name:
                parts = exp_name.split("_")
                for part in parts:
                    if part.startswith("sub"):
                        # Remove bracketed timestamp if present
                        subject = part.split()[0] if " " in part else part
                        break
            
            if not subject:
                continue
            
            # Look for run directory (typically run 1)
            run_dir = exp_dir / "1"
            if not run_dir.exists():
                continue
            
            validations_path = run_dir / "validations.json"
            if not validations_path.exists():
                continue
            
            with open(validations_path, 'r') as f:
                validations = json.load(f)
            
            if not validations:
                continue
            
            # Extract epoch-wise data
            epochs = []
            lls = []
            nmis = []
            
            for epoch_str in sorted(validations.keys(), key=lambda x: int(x)):
                epoch = int(epoch_str)
                val = validations[epoch_str]
                
                ll = val.get('log_likelihood')
                nmi = val.get('nmi')
                
                # Only include epochs with at least one metric
                if ll is not None or nmi is not None:
                    epochs.append(epoch)
                    lls.append(ll)
                    nmis.append(nmi)
            
            if epochs:
                data_by_model[current_model][subject] = {
                    'epochs': epochs,
                    'll': lls,
                    'nmi': nmis
                }
    
    if not data_by_model:
        print("⚠️  No epoch progression data found")
        return generated_plots
    
    # Generate progression plots for each model type
    for model, subjects_data in data_by_model.items():
        if not subjects_data:
            continue
        
        subjects = sorted(subjects_data.keys())
        colors = sns.color_palette("husl", len(subjects))
        
        # Create combined plot with all subjects
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        sns.despine()
        
        # Plot Log Likelihood progression
        for idx, subject in enumerate(subjects):
            data = subjects_data[subject]
            epochs = data['epochs']
            lls = data['ll']
            
            # Filter out None values
            valid_data = [(e, ll) for e, ll in zip(epochs, lls) if ll is not None]
            if valid_data:
                valid_epochs, valid_lls = zip(*valid_data)
                ax1.plot(valid_epochs, valid_lls, marker='o', markersize=4,
                        linewidth=2, label=subject, color=colors[idx], alpha=0.8)
        
        ax1.set_xlabel('Training Epoch', fontsize=11)
        ax1.set_ylabel('Validation Log Likelihood', fontsize=11)
        ax1.set_title(f'{model.upper()}: LL Progression Over Training', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot NMI progression
        for idx, subject in enumerate(subjects):
            data = subjects_data[subject]
            epochs = data['epochs']
            nmis = data['nmi']
            
            # Filter out None values
            valid_data = [(e, nmi) for e, nmi in zip(epochs, nmis) if nmi is not None]
            if valid_data:
                valid_epochs, valid_nmis = zip(*valid_data)
                ax2.plot(valid_epochs, valid_nmis, marker='o', markersize=4,
                        linewidth=2, label=subject, color=colors[idx], alpha=0.8)
        
        ax2.set_xlabel('Training Epoch', fontsize=11)
        ax2.set_ylabel('NMI vs True States', fontsize=11)
        ax2.set_title(f'{model.upper()}: NMI Progression Over Training', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle(f'Validation Metrics Progression During Training ({model.upper()})',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save plot
        output_file = gen_dir / model / f"epoch_progression_{model}.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        generated_plots.append(output_file)
        print(f"✓ Generated epoch progression: {output_file.relative_to(gen_dir)}")
    
    return generated_plots


def generate_multi_subject_comparison(gen_dir: Path, model_type: str = None) -> list[Path]:
    """Generate comparison plot across all test subjects for a model type.
    
    Args:
        gen_dir: Root generalization directory
        model_type: Specific model type (hmm/marhmm) or None for all
        
    Returns:
        List of generated plot paths
    """
    # Collect data by model type and subject
    data_by_model = defaultdict(dict)  # {model_type: {subject: {'ll': ..., 'nmi': ...}}}
    generated_plots = []
    
    if not gen_dir.exists():
        return generated_plots
    
    # Scan model type subdirectories
    for model_dir in gen_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        current_model = model_dir.name
        if model_type and current_model != model_type:
            continue
        
        # Scan experiment directories
        for exp_dir in model_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Extract subject from directory name
            exp_name = exp_dir.name
            subject = None
            if "sub" in exp_name:
                parts = exp_name.split("_")
                for part in parts:
                    if part.startswith("sub"):
                        # Remove bracketed timestamp if present
                        subject = part.split()[0] if " " in part else part
                        break
            
            if not subject:
                continue
            
            # Look for run directory (typically run 1)
            run_dir = exp_dir / "1"
            if not run_dir.exists():
                continue
            
            validations_path = run_dir / "validations.json"
            if not validations_path.exists():
                continue
            
            with open(validations_path, 'r') as f:
                validations = json.load(f)
            
            if not validations:
                continue
            
            # Extract key time points: start, NMI peak, and end
            epochs_sorted = sorted([int(k) for k in validations.keys()])
            
            # Find start epoch (first with LL)
            start_ll = None
            start_nmi = None
            start_epoch = None
            for epoch in epochs_sorted:
                val = validations[str(epoch)]
                if val.get('log_likelihood') is not None:
                    start_ll = val.get('log_likelihood')
                    start_nmi = val.get('nmi')  # Also get NMI at this epoch (may be None)
                    start_epoch = epoch
                    break
            
            # Find NMI peak epoch
            peak_nmi_epoch = None
            peak_nmi_value = -1
            peak_ll = None
            for epoch in epochs_sorted:
                val = validations[str(epoch)]
                nmi = val.get('nmi')
                if nmi is not None and nmi > peak_nmi_value:
                    peak_nmi_value = nmi
                    peak_nmi_epoch = epoch
                    # Get LL at this epoch (might be None)
                    peak_ll = val.get('log_likelihood')
            
            # Find end epoch (last with LL)
            end_ll = None
            end_epoch = None
            end_nmi = None
            for epoch in reversed(epochs_sorted):
                val = validations[str(epoch)]
                if val.get('log_likelihood') is not None:
                    end_ll = val.get('log_likelihood')
                    end_epoch = epoch
                    end_nmi = val.get('nmi')
                    break
            
            # If no LL at peak, try to find nearest LL
            if peak_ll is None and peak_nmi_epoch is not None:
                # Search nearby epochs for LL
                for offset in range(1, 10):
                    for direction in [-1, 1]:
                        nearby_epoch = peak_nmi_epoch + (offset * direction)
                        if str(nearby_epoch) in validations:
                            nearby_ll = validations[str(nearby_epoch)].get('log_likelihood')
                            if nearby_ll is not None:
                                peak_ll = nearby_ll
                                break
                    if peak_ll is not None:
                        break
            
            if start_ll is not None or peak_ll is not None or end_ll is not None:
                data_by_model[current_model][subject] = {
                    'start_ll': start_ll,
                    'start_nmi': start_nmi,
                    'start_epoch': start_epoch,
                    'peak_ll': peak_ll,
                    'peak_nmi': peak_nmi_value if peak_nmi_value > 0 else None,
                    'peak_epoch': peak_nmi_epoch,
                    'end_ll': end_ll,
                    'end_nmi': end_nmi,
                    'end_epoch': end_epoch
                }
    
    if not data_by_model:
        print("⚠️  No multi-subject data found")
        return generated_plots
    
    # Generate comparison plots for each model type
    for model, subjects_data in data_by_model.items():
        if len(subjects_data) < 2:
            continue
        
        subjects = sorted(subjects_data.keys())
        colors = sns.color_palette("husl", len(subjects))
        
        # Create comparison plot showing LL at three key time points
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sns.despine()
        
        # LL Progression: Start -> Peak NMI -> End
        time_points = ['Start', 'NMI Peak', 'End']
        x_pos = np.arange(len(time_points))
        
        for idx, subject in enumerate(subjects):
            data = subjects_data[subject]
            ll_values = [data['start_ll'], data['peak_ll'], data['end_ll']]
            epochs = [data['start_epoch'], data['peak_epoch'], data['end_epoch']]
            
            # Filter out None values but keep track of positions
            valid_points = [(i, ll, ep) for i, (ll, ep) in enumerate(zip(ll_values, epochs)) if ll is not None]
            
            if valid_points:
                indices, values, epoch_labels = zip(*valid_points)
                ax1.plot([x_pos[i] for i in indices], values,
                        marker='o', markersize=10, linewidth=2.5,
                        label=subject, color=colors[idx], alpha=0.8,
                        markeredgecolor='black', markeredgewidth=1.5)
                
                # Add value labels
                for i, (idx_val, val, ep) in enumerate(valid_points):
                    ax1.text(x_pos[idx_val], val, f'{val:.4f}',
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     edgecolor='black', alpha=0.7))
        
        ax1.set_ylabel('Validation Log Likelihood', fontsize=11)
        ax1.set_title(f'{model.upper()}: LL at Key Training Points', fontsize=12, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(time_points, fontsize=10)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(axis='both', alpha=0.3, linestyle='--')
        
        # NMI Progression: Start -> Peak -> End  
        for idx, subject in enumerate(subjects):
            data = subjects_data[subject]
            # Get NMI at all three time points
            nmi_start = data['start_nmi']
            nmi_peak = data['peak_nmi']
            nmi_end = data['end_nmi']
            
            # Plot NMI at all three time points
            valid_nmi_points = []
            if nmi_start is not None:
                valid_nmi_points.append((0, nmi_start))  # Start is at position 0
            if nmi_peak is not None:
                valid_nmi_points.append((1, nmi_peak))  # Peak is at position 1
            if nmi_end is not None:
                valid_nmi_points.append((2, nmi_end))  # End is at position 2
            
            if valid_nmi_points:
                indices, values = zip(*valid_nmi_points)
                ax2.plot([x_pos[i] for i in indices], values,
                        marker='o', markersize=10, linewidth=2.5,
                        label=subject, color=colors[idx], alpha=0.8,
                        markeredgecolor='black', markeredgewidth=1.5)
                
                # Add value labels
                for idx_val, val in valid_nmi_points:
                    ax2.text(x_pos[idx_val], val + 0.03, f'{val:.3f}',
                            ha='center', va='bottom', fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                     edgecolor='black', alpha=0.7))
        
        ax2.set_ylabel('NMI vs True States', fontsize=11)
        ax2.set_title(f'{model.upper()}: NMI at Key Training Points', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(time_points, fontsize=10)
        ax2.set_ylim([0, 1])
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(axis='both', alpha=0.3, linestyle='--')
        
        plt.suptitle(f'Generalization Performance Across Test Subjects ({model.upper()})', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save plot
        output_file = gen_dir / model / f"comparison_across_subjects_{model}.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        generated_plots.append(output_file)
        print(f"✓ Generated multi-subject comparison: {output_file.relative_to(gen_dir)}")
    
    return generated_plots


def main():
    parser = argparse.ArgumentParser(
        description="Generate missing validation performance plots for generalization experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate individual plots only
  python scripts/generate_generalization_plots.py results/generalization/
  
  # Generate with multi-subject comparison
  python scripts/generate_generalization_plots.py results/generalization/ --comparison
  
  # Generate with epoch progression plots
  python scripts/generate_generalization_plots.py results/generalization/ --progression
  
  # Generate all plot types
  python scripts/generate_generalization_plots.py results/generalization/ --comparison --progression
"""
    )
    
    parser.add_argument("gen_dir", type=str,
                       help="Path to generalization results directory")
    parser.add_argument("--comparison", action="store_true",
                       help="Also generate multi-subject comparison plots")
    parser.add_argument("--progression", action="store_true",
                       help="Also generate epoch progression plots showing metrics over training")
    
    args = parser.parse_args()
    
    gen_dir = Path(args.gen_dir)
    
    print(f"\n{'='*70}")
    print("GENERALIZATION PLOT GENERATOR")
    print(f"{'='*70}")
    print(f"Directory: {gen_dir}")
    print(f"{'='*70}\n")
    
    # Find all runs
    print("📂 Scanning for generalization runs...")
    run_dirs = find_generalization_runs(gen_dir)
    
    if not run_dirs:
        print(f"❌ No generalization runs found in {gen_dir}")
        sys.exit(1)
    
    print(f"✓ Found {len(run_dirs)} runs\n")
    
    # Generate plots
    print("🎨 Generating plots (forcing regeneration)...\n")
    generated = 0
    failed = 0
    generated_plots = []  # Track paths for display
    
    for run_dir in run_dirs:
        relative_path = run_dir.relative_to(gen_dir)
        try:
            success, plot_path = generate_validation_plot(run_dir)
            if success:
                print(f"✓ Generated: {relative_path}")
                generated += 1
                if plot_path:
                    generated_plots.append(plot_path)
        except Exception as e:
            print(f"❌ Failed: {relative_path} - {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Generated: {generated}")
    print(f"Failed: {failed}")
    print(f"{'='*70}\n")
    
    if generated > 0:
        print(f"✅ Successfully regenerated {generated} validation performance plots!")
        print(f"\n📂 INDIVIDUAL PLOTS SAVED TO:")
        for plot_path in generated_plots[:10]:  # Show first 10
            print(f"   {plot_path}")
        if len(generated_plots) > 10:
            print(f"   ... and {len(generated_plots) - 10} more")
    elif failed > 0:
        print(f"⚠️  Failed to generate plots (see errors above)")
    else:
        print("⚠️  No plots were generated")
    
    # Generate multi-subject comparison if requested
    if args.comparison:
        print(f"\n{'='*70}")
        print("GENERATING MULTI-SUBJECT COMPARISONS")
        print(f"{'='*70}\n")
        comparison_plots = generate_multi_subject_comparison(gen_dir)
        if comparison_plots:
            print(f"\n📊 MULTI-SUBJECT COMPARISON PLOTS:")
            for plot_path in comparison_plots:
                print(f"   {plot_path}")
    
    # Generate epoch progression plots if requested
    if args.progression:
        print(f"\n{'='*70}")
        print("GENERATING EPOCH PROGRESSION PLOTS")
        print(f"{'='*70}\n")
        progression_plots = generate_epoch_progression_plots(gen_dir)
        if progression_plots:
            print(f"\n📈 EPOCH PROGRESSION PLOTS:")
            for plot_path in progression_plots:
                print(f"   {plot_path}")


if __name__ == "__main__":
    main()
