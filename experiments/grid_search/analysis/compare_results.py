"""
Grid Search Results Analysis
============================
Generates comparison plots and identifies best hyperparameters.

Outputs:
- Bar charts comparing R², MAE, RMSE, Pearson across all experiments
- Separate visualizations for Cosine Annealing vs One Cycle
- Best configuration summary
- Optional: validation curve overlays
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_results(summary_csv: str) -> List[Dict[str, Any]]:
    """Load results from summary CSV."""
    results = []
    with open(summary_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in ['r2', 'mae', 'mse', 'rmse', 'pearson', 'median_ae', 
                       'mean_error', 'std_error', 'duration_s']:
                if key in row and row[key]:
                    row[key] = float(row[key])
            for key in ['epochs_trained', 'best_epoch']:
                if key in row and row[key]:
                    row[key] = int(row[key])
            results.append(row)
    return results


def load_validation_history(results_dir: str, exp_name: str) -> Optional[Dict]:
    """Load validation history for an experiment."""
    history_path = os.path.join(results_dir, exp_name, 'validation_history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"  Warning: Corrupted validation_history.json for {exp_name}, skipping")
            return None
    return None


def plot_metric_comparison(
    results: List[Dict],
    metric: str,
    title: str,
    ylabel: str,
    save_path: str,
    higher_is_better: bool = True
):
    """Plot bar chart comparing a metric across all experiments."""
    
    # Separate by scheduler type
    cosine_results = [r for r in results if r['scheduler'] == 'cosine_annealing']
    onecycle_results = [r for r in results if r['scheduler'] == 'one_cycle']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colors_cosine = plt.cm.Blues(np.linspace(0.4, 0.8, len(cosine_results)))
    colors_onecycle = plt.cm.Oranges(np.linspace(0.4, 0.8, len(onecycle_results)))
    
    # Plot Cosine Annealing
    if cosine_results:
        names = [r['experiment'].replace('cosine_', '').replace('_', '\n') 
                for r in cosine_results]
        values = [r[metric] for r in cosine_results]
        
        bars = ax1.bar(range(len(values)), values, color=colors_cosine)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, fontsize=8)
        ax1.set_ylabel(ylabel)
        ax1.set_title('Cosine Annealing Scheduler')
        ax1.grid(axis='y', alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(values) if higher_is_better else np.argmin(values)
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)
        
        # Add value labels
        for i, v in enumerate(values):
            ax1.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot One Cycle
    if onecycle_results:
        names = [r['experiment'].replace('onecycle_', '').replace('_', '\n') 
                for r in onecycle_results]
        values = [r[metric] for r in onecycle_results]
        
        bars = ax2.bar(range(len(values)), values, color=colors_onecycle)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, fontsize=8)
        ax2.set_ylabel(ylabel)
        ax2.set_title('One Cycle Scheduler')
        ax2.grid(axis='y', alpha=0.3)
        
        # Highlight best
        best_idx = np.argmax(values) if higher_is_better else np.argmin(values)
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)
        
        # Add value labels
        for i, v in enumerate(values):
            ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Equalize y-axis
    all_values = [r[metric] for r in results]
    y_min = min(all_values) * 0.95
    y_max = max(all_values) * 1.05
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_combined_comparison(
    results: List[Dict],
    save_path: str
):
    """Plot all key metrics in a single figure."""
    
    metrics = [
        ('r2', 'R² Score', True),
        ('mae', 'MAE', False),
        ('rmse', 'RMSE', False),
        ('pearson', 'Pearson Correlation', True)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Sort results: cosine first, then onecycle
    cosine = sorted([r for r in results if r['scheduler'] == 'cosine_annealing'],
                   key=lambda x: x['experiment'])
    onecycle = sorted([r for r in results if r['scheduler'] == 'one_cycle'],
                     key=lambda x: x['experiment'])
    sorted_results = cosine + onecycle
    
    names = []
    for r in sorted_results:
        name = r['experiment']
        name = name.replace('cosine_', 'C:').replace('onecycle_', 'O:')
        name = name.replace('lr', '').replace('min', 'm').replace('max', '')
        name = name.replace('pct', 'p')
        names.append(name)
    
    colors = ['steelblue'] * len(cosine) + ['darkorange'] * len(onecycle)
    
    for ax, (metric, label, higher_better) in zip(axes, metrics):
        values = [r[metric] for r in sorted_results]
        
        bars = ax.bar(range(len(values)), values, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis='y', alpha=0.3)
        
        # Highlight overall best
        best_idx = np.argmax(values) if higher_better else np.argmin(values)
        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)
    
    # Add legend
    cosine_patch = mpatches.Patch(color='steelblue', label='Cosine Annealing')
    onecycle_patch = mpatches.Patch(color='darkorange', label='One Cycle')
    fig.legend(handles=[cosine_patch, onecycle_patch], loc='upper right', fontsize=10)
    
    plt.suptitle('Grid Search Results: Scheduler Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_validation_curves(
    results: List[Dict],
    results_dir: str,
    save_path: str
):
    """Plot validation loss curves for all experiments."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Cosine Annealing curves
    cosine_results = [r for r in results if r['scheduler'] == 'cosine_annealing']
    colors_cosine = plt.cm.Blues(np.linspace(0.3, 0.9, len(cosine_results)))
    
    for i, r in enumerate(cosine_results):
        history = load_validation_history(results_dir, r['experiment'])
        if history:
            label = r['experiment'].replace('cosine_', '')
            ax1.plot(history['val_losses'], color=colors_cosine[i], 
                    label=label, alpha=0.8)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Cosine Annealing - Validation Loss')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.grid(alpha=0.3)
    
    # One Cycle curves
    onecycle_results = [r for r in results if r['scheduler'] == 'one_cycle']
    colors_onecycle = plt.cm.Oranges(np.linspace(0.3, 0.9, len(onecycle_results)))
    
    for i, r in enumerate(onecycle_results):
        history = load_validation_history(results_dir, r['experiment'])
        if history:
            label = r['experiment'].replace('onecycle_', '')
            ax2.plot(history['val_losses'], color=colors_onecycle[i],
                    label=label, alpha=0.8)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('One Cycle - Validation Loss')
    ax2.legend(fontsize=7, loc='upper right')
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Validation Loss Curves Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_heatmaps(results: List[Dict], save_path: str):
    """Plot heatmaps showing parameter sensitivity for each scheduler."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Cosine Annealing heatmap (R² score)
    cosine_results = [r for r in results if r['scheduler'] == 'cosine_annealing']
    if cosine_results:
        # Extract unique parameter values
        import json as json_module
        lrs = sorted(set(json_module.loads(r['params'])['lr'] for r in cosine_results))
        min_lrs = sorted(set(json_module.loads(r['params'])['min_lr'] for r in cosine_results))
        
        # Create R² matrix
        r2_matrix = np.zeros((len(lrs), len(min_lrs)))
        for r in cosine_results:
            params = json_module.loads(r['params'])
            i = lrs.index(params['lr'])
            j = min_lrs.index(params['min_lr'])
            r2_matrix[i, j] = r['r2']
        
        im1 = axes[0, 0].imshow(r2_matrix, cmap='RdYlGn', aspect='auto')
        axes[0, 0].set_xticks(range(len(min_lrs)))
        axes[0, 0].set_xticklabels([f'{x:.0e}' for x in min_lrs])
        axes[0, 0].set_yticks(range(len(lrs)))
        axes[0, 0].set_yticklabels([str(x) for x in lrs])
        axes[0, 0].set_xlabel('min_lr')
        axes[0, 0].set_ylabel('learning_rate')
        axes[0, 0].set_title('Cosine Annealing - R² Score')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Add text annotations
        for i in range(len(lrs)):
            for j in range(len(min_lrs)):
                axes[0, 0].text(j, i, f'{r2_matrix[i, j]:.3f}', 
                               ha='center', va='center', fontsize=9)
        
        # MAE matrix
        mae_matrix = np.zeros((len(lrs), len(min_lrs)))
        for r in cosine_results:
            params = json_module.loads(r['params'])
            i = lrs.index(params['lr'])
            j = min_lrs.index(params['min_lr'])
            mae_matrix[i, j] = r['mae']
        
        im2 = axes[0, 1].imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto')
        axes[0, 1].set_xticks(range(len(min_lrs)))
        axes[0, 1].set_xticklabels([f'{x:.0e}' for x in min_lrs])
        axes[0, 1].set_yticks(range(len(lrs)))
        axes[0, 1].set_yticklabels([str(x) for x in lrs])
        axes[0, 1].set_xlabel('min_lr')
        axes[0, 1].set_ylabel('learning_rate')
        axes[0, 1].set_title('Cosine Annealing - MAE (lower is better)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        for i in range(len(lrs)):
            for j in range(len(min_lrs)):
                axes[0, 1].text(j, i, f'{mae_matrix[i, j]:.3f}', 
                               ha='center', va='center', fontsize=9)
    
    # One Cycle heatmap
    onecycle_results = [r for r in results if r['scheduler'] == 'one_cycle']
    if onecycle_results:
        import json as json_module
        max_lrs = sorted(set(json_module.loads(r['params'])['max_lr'] for r in onecycle_results))
        pct_starts = sorted(set(json_module.loads(r['params'])['pct_start'] for r in onecycle_results))
        
        # R² matrix
        r2_matrix = np.zeros((len(max_lrs), len(pct_starts)))
        for r in onecycle_results:
            params = json_module.loads(r['params'])
            i = max_lrs.index(params['max_lr'])
            j = pct_starts.index(params['pct_start'])
            r2_matrix[i, j] = r['r2']
        
        im3 = axes[1, 0].imshow(r2_matrix, cmap='RdYlGn', aspect='auto')
        axes[1, 0].set_xticks(range(len(pct_starts)))
        axes[1, 0].set_xticklabels([str(x) for x in pct_starts])
        axes[1, 0].set_yticks(range(len(max_lrs)))
        axes[1, 0].set_yticklabels([str(x) for x in max_lrs])
        axes[1, 0].set_xlabel('pct_start')
        axes[1, 0].set_ylabel('max_lr')
        axes[1, 0].set_title('One Cycle - R² Score')
        plt.colorbar(im3, ax=axes[1, 0])
        
        for i in range(len(max_lrs)):
            for j in range(len(pct_starts)):
                axes[1, 0].text(j, i, f'{r2_matrix[i, j]:.3f}', 
                               ha='center', va='center', fontsize=9)
        
        # MAE matrix
        mae_matrix = np.zeros((len(max_lrs), len(pct_starts)))
        for r in onecycle_results:
            params = json_module.loads(r['params'])
            i = max_lrs.index(params['max_lr'])
            j = pct_starts.index(params['pct_start'])
            mae_matrix[i, j] = r['mae']
        
        im4 = axes[1, 1].imshow(mae_matrix, cmap='RdYlGn_r', aspect='auto')
        axes[1, 1].set_xticks(range(len(pct_starts)))
        axes[1, 1].set_xticklabels([str(x) for x in pct_starts])
        axes[1, 1].set_yticks(range(len(max_lrs)))
        axes[1, 1].set_yticklabels([str(x) for x in max_lrs])
        axes[1, 1].set_xlabel('pct_start')
        axes[1, 1].set_ylabel('max_lr')
        axes[1, 1].set_title('One Cycle - MAE (lower is better)')
        plt.colorbar(im4, ax=axes[1, 1])
        
        for i in range(len(max_lrs)):
            for j in range(len(pct_starts)):
                axes[1, 1].text(j, i, f'{mae_matrix[i, j]:.3f}', 
                               ha='center', va='center', fontsize=9)
    
    plt.suptitle('Parameter Sensitivity Heatmaps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def print_best_results(results: List[Dict]):
    """Print best configurations for each metric."""
    
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS PER METRIC")
    print("="*70)
    
    metrics = [
        ('r2', 'R² Score', True),
        ('mae', 'MAE', False),
        ('rmse', 'RMSE', False),
        ('pearson', 'Pearson Correlation', True),
        ('median_ae', 'Median Absolute Error', False)
    ]
    
    for metric, label, higher_better in metrics:
        if higher_better:
            best = max(results, key=lambda x: x[metric])
        else:
            best = min(results, key=lambda x: x[metric])
        
        print(f"\nBest {label}: {best[metric]:.4f}")
        print(f"  Experiment: {best['experiment']}")
        print(f"  Scheduler:  {best['scheduler']}")
        print(f"  Params:     {best['params']}")
    
    # Overall recommendation
    print("\n" + "="*70)
    print("🏆 OVERALL WINNER DECLARATION 🏆")
    print("="*70)
    
    # Score each experiment (normalize and combine metrics)
    for r in results:
        r2_values = [x['r2'] for x in results]
        mae_values = [x['mae'] for x in results]
        rmse_values = [x['rmse'] for x in results]
        pearson_values = [x['pearson'] for x in results]
        
        # Normalize: higher R²/Pearson is better, lower MAE/RMSE is better
        r2_range = max(r2_values) - min(r2_values) + 1e-8
        mae_range = max(mae_values) - min(mae_values) + 1e-8
        rmse_range = max(rmse_values) - min(rmse_values) + 1e-8
        pearson_range = max(pearson_values) - min(pearson_values) + 1e-8
        
        r['score'] = (
            (r['r2'] - min(r2_values)) / r2_range * 0.3 +
            (max(mae_values) - r['mae']) / mae_range * 0.25 +
            (max(rmse_values) - r['rmse']) / rmse_range * 0.25 +
            (r['pearson'] - min(pearson_values)) / pearson_range * 0.2
        )
    
    # Sort by score
    ranked = sorted(results, key=lambda x: x['score'], reverse=True)
    best_overall = ranked[0]
    
    print(f"\n  🥇 WINNER: {best_overall['experiment']}")
    print(f"  " + "-"*50)
    print(f"  Scheduler:     {best_overall['scheduler']}")
    print(f"  Parameters:    {best_overall['params']}")
    print(f"  " + "-"*50)
    print(f"  R² Score:      {best_overall['r2']:.4f}")
    print(f"  MAE:           {best_overall['mae']:.4f}")
    print(f"  RMSE:          {best_overall['rmse']:.4f}")
    print(f"  Pearson:       {best_overall['pearson']:.4f}")
    print(f"  Median AE:     {best_overall['median_ae']:.4f}")
    print(f"  " + "-"*50)
    print(f"  Composite Score: {best_overall['score']:.4f}")
    
    # Runner ups
    if len(ranked) >= 3:
        print(f"\n  🥈 Runner-up:  {ranked[1]['experiment']} (score: {ranked[1]['score']:.4f})")
        print(f"  🥉 Third:      {ranked[2]['experiment']} (score: {ranked[2]['score']:.4f})")
    
    # Compare scheduler types
    print("\n" + "="*70)
    print("SCHEDULER COMPARISON (Average ± Std)")
    print("="*70)
    
    cosine = [r for r in results if r['scheduler'] == 'cosine_annealing']
    onecycle = [r for r in results if r['scheduler'] == 'one_cycle']
    
    if cosine:
        print(f"\nCosine Annealing (n={len(cosine)}):")
        print(f"  R²:        {np.mean([r['r2'] for r in cosine]):.4f} ± {np.std([r['r2'] for r in cosine]):.4f}")
        print(f"  MAE:       {np.mean([r['mae'] for r in cosine]):.4f} ± {np.std([r['mae'] for r in cosine]):.4f}")
        print(f"  RMSE:      {np.mean([r['rmse'] for r in cosine]):.4f} ± {np.std([r['rmse'] for r in cosine]):.4f}")
        print(f"  Pearson:   {np.mean([r['pearson'] for r in cosine]):.4f} ± {np.std([r['pearson'] for r in cosine]):.4f}")
        print(f"  Avg Score: {np.mean([r['score'] for r in cosine]):.4f}")
    
    if onecycle:
        print(f"\nOne Cycle (n={len(onecycle)}):")
        print(f"  R²:        {np.mean([r['r2'] for r in onecycle]):.4f} ± {np.std([r['r2'] for r in onecycle]):.4f}")
        print(f"  MAE:       {np.mean([r['mae'] for r in onecycle]):.4f} ± {np.std([r['mae'] for r in onecycle]):.4f}")
        print(f"  RMSE:      {np.mean([r['rmse'] for r in onecycle]):.4f} ± {np.std([r['rmse'] for r in onecycle]):.4f}")
        print(f"  Pearson:   {np.mean([r['pearson'] for r in onecycle]):.4f} ± {np.std([r['pearson'] for r in onecycle]):.4f}")
        print(f"  Avg Score: {np.mean([r['score'] for r in onecycle]):.4f}")
    
    # Declare scheduler winner
    if cosine and onecycle:
        cosine_avg = np.mean([r['score'] for r in cosine])
        onecycle_avg = np.mean([r['score'] for r in onecycle])
        winner_scheduler = "Cosine Annealing" if cosine_avg > onecycle_avg else "One Cycle"
        margin = abs(cosine_avg - onecycle_avg)
        print(f"\n  → Best Scheduler Overall: {winner_scheduler} (margin: {margin:.4f})")


def generate_summary_table(results: List[Dict], save_path: str):
    """Generate a formatted summary table."""
    
    # Sort by R² descending
    sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)
    
    with open(save_path, 'w') as f:
        f.write("Grid Search Results Summary\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"{'Rank':<5} {'Experiment':<35} {'R²':<8} {'MAE':<8} {'RMSE':<8} {'Pearson':<8} {'Time(s)':<8}\n")
        f.write("-"*100 + "\n")
        
        for i, r in enumerate(sorted_results, 1):
            f.write(f"{i:<5} {r['experiment']:<35} {r['r2']:<8.4f} {r['mae']:<8.4f} "
                   f"{r['rmse']:<8.4f} {r['pearson']:<8.4f} {r['duration_s']:<8.1f}\n")
        
        f.write("\n")
    
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Grid Search Results')
    parser.add_argument('--results-dir', type=str,
                       default='experiments/grid_search/results',
                       help='Results directory')
    parser.add_argument('--summary-csv', type=str,
                       default='experiments/grid_search/results/summary.csv',
                       help='Summary CSV file')
    parser.add_argument('--output-dir', type=str,
                       default='experiments/grid_search/analysis',
                       help='Output directory for plots')
    args = parser.parse_args()
    
    # Check if results exist
    if not os.path.exists(args.summary_csv):
        print(f"Error: Results file not found: {args.summary_csv}")
        print("Run the grid search first:")
        print("  python experiments/grid_search/run_grid_search.py")
        return
    
    # Load results
    results = load_results(args.summary_csv)
    print(f"Loaded {len(results)} experiment results")
    
    if len(results) == 0:
        print("No results to analyze")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # Individual metric plots
    plot_metric_comparison(
        results, 'r2', 'R² Score Comparison',
        'R² Score', os.path.join(args.output_dir, 'r2_comparison.png'),
        higher_is_better=True
    )
    
    plot_metric_comparison(
        results, 'mae', 'MAE Comparison',
        'Mean Absolute Error', os.path.join(args.output_dir, 'mae_comparison.png'),
        higher_is_better=False
    )
    
    plot_metric_comparison(
        results, 'rmse', 'RMSE Comparison',
        'Root Mean Squared Error', os.path.join(args.output_dir, 'rmse_comparison.png'),
        higher_is_better=False
    )
    
    plot_metric_comparison(
        results, 'pearson', 'Pearson Correlation Comparison',
        'Pearson Correlation', os.path.join(args.output_dir, 'pearson_comparison.png'),
        higher_is_better=True
    )
    
    plot_metric_comparison(
        results, 'median_ae', 'Median Absolute Error Comparison',
        'Median Absolute Error', os.path.join(args.output_dir, 'median_ae_comparison.png'),
        higher_is_better=False
    )
    
    # Combined comparison
    plot_combined_comparison(
        results,
        os.path.join(args.output_dir, 'combined_comparison.png')
    )
    
    # Parameter sensitivity heatmaps
    plot_heatmaps(
        results,
        os.path.join(args.output_dir, 'parameter_heatmaps.png')
    )
    
    # Validation curves
    plot_validation_curves(
        results,
        args.results_dir,
        os.path.join(args.output_dir, 'validation_curves.png')
    )
    
    # Summary table
    generate_summary_table(
        results,
        os.path.join(args.output_dir, 'summary_table.txt')
    )
    
    # Print best results
    print_best_results(results)
    
    print("\n" + "="*70)
    print(f"Analysis complete! Plots saved to: {args.output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
