"""
Compare results from edge ablation experiments.
Loads metrics from normal, self_loops, and zero_features modes and generates comparative analysis.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def load_results(results_dir):
    """Load results from all three modes."""
    modes = ['normal', 'self_loops', 'zero_features']
    results = {}

    for mode in modes:
        metrics_path = os.path.join(results_dir, mode, 'test_metrics.json')
        if not os.path.exists(metrics_path):
            print(f"Warning: Results not found for mode '{mode}' at {metrics_path}")
            continue

        with open(metrics_path, 'r') as f:
            data = json.load(f)
            results[mode] = data

    return results


def compute_degradation(baseline_value, test_value):
    """Compute percentage degradation from baseline."""
    if baseline_value == 0:
        return 0.0
    return ((test_value - baseline_value) / abs(baseline_value)) * 100


def print_comparison_table(results):
    """Print a formatted comparison table."""
    if 'normal' not in results:
        print("Error: Normal baseline results not found")
        return

    modes = ['normal', 'self_loops', 'zero_features']
    mode_names = {
        'normal': 'Normal (Baseline)',
        'self_loops': 'Self-Loops Only',
        'zero_features': 'Zero Edge Features'
    }

    print("\n" + "=" * 100)
    print("EDGE ABLATION STUDY - RESULTS COMPARISON")
    print("=" * 100)

    # Metrics to compare
    metrics = ['mae', 'rmse', 'r2', 'pearson_corr', 'median_ae']
    metric_names = {
        'mae': 'MAE',
        'rmse': 'RMSE',
        'r2': 'R²',
        'pearson_corr': 'Pearson Corr',
        'median_ae': 'Median AE'
    }

    # Header
    print(f"\n{'Metric':<20} {'Normal':<15} {'Self-Loops':<20} {'Zero Features':<20}")
    print("-" * 100)

    # Get baseline values
    baseline = results['normal']['metrics']

    # Print each metric
    for metric in metrics:
        values = []
        for mode in modes:
            if mode in results:
                val = results[mode]['metrics'][metric]
                values.append(f"{val:.4f}")

                # Add degradation for non-baseline modes
                if mode != 'normal' and metric != 'r2':
                    deg = compute_degradation(baseline[metric], val)
                    values[-1] += f" ({deg:+.1f}%)"
                elif mode != 'normal' and metric == 'r2':
                    # For R², show absolute difference
                    diff = val - baseline[metric]
                    values[-1] += f" ({diff:+.4f})"
            else:
                values.append("N/A")

        print(f"{metric_names[metric]:<20} {values[0]:<15} {values[1]:<20} {values[2]:<20}")

    print("=" * 100)

    # Key findings
    print("\nKEY FINDINGS:")
    print("-" * 100)

    if 'self_loops' in results:
        r2_normal = baseline['r2']
        r2_self_loops = results['self_loops']['metrics']['r2']
        r2_diff = r2_self_loops - r2_normal

        if abs(r2_diff) < 0.05:
            print(f"1. Graph structure contributes MINIMALLY (R² change: {r2_diff:+.4f})")
            print("   → Model primarily uses node features, not graph connectivity")
        else:
            print(f"1. Graph structure contributes SIGNIFICANTLY (R² change: {r2_diff:+.4f})")
            print("   → Model leverages graph connectivity for predictions")

    if 'zero_features' in results:
        r2_normal = baseline['r2']
        r2_zero = results['zero_features']['metrics']['r2']
        r2_diff = r2_zero - r2_normal

        if abs(r2_diff) < 0.05:
            print(f"\n2. Edge features contribute MINIMALLY (R² change: {r2_diff:+.4f})")
            print("   → Model doesn't use edge features (bond types, distances, etc.)")
        else:
            print(f"\n2. Edge features contribute SIGNIFICANTLY (R² change: {r2_diff:+.4f})")
            print("   → Model leverages edge features for predictions")

    print("=" * 100 + "\n")


def create_comparison_plots(results, output_dir):
    """Create comparison visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    modes = ['normal', 'self_loops', 'zero_features']
    mode_labels = {
        'normal': 'Normal',
        'self_loops': 'Self-Loops',
        'zero_features': 'Zero Features'
    }

    # Filter available modes
    available_modes = [m for m in modes if m in results]

    if len(available_modes) < 2:
        print("Not enough results to create comparison plots")
        return

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3)

    # Metrics to plot
    metrics = ['mae', 'rmse', 'r2']
    metric_names = ['MAE', 'RMSE', 'R²']

    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = fig.add_subplot(gs[0, idx])

        values = [results[mode]['metrics'][metric] for mode in available_modes]
        labels = [mode_labels[mode] for mode in available_modes]
        colors = ['#2ecc71', '#e74c3c', '#3498db'][:len(available_modes)]

        bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{val:.4f}',
                ha='center',
                va='bottom',
                fontsize=10,
                fontweight='bold'
            )

        ax.set_ylabel(name, fontsize=12, fontweight='bold')
        ax.set_title(f'{name} Comparison', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(values) * 1.15)

    plt.suptitle(
        'Edge Ablation Study - Metrics Comparison',
        fontsize=15,
        fontweight='bold',
        y=1.02
    )

    save_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.close()


def save_comparison_json(results, output_dir):
    """Save comparison results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    if 'normal' not in results:
        print("Error: Normal baseline results not found")
        return

    baseline = results['normal']['metrics']
    comparison = {
        'baseline': baseline,
        'ablations': {}
    }

    for mode in ['self_loops', 'zero_features']:
        if mode in results:
            metrics = results[mode]['metrics']
            comparison['ablations'][mode] = {
                'metrics': metrics,
                'degradation': {
                    'mae': compute_degradation(baseline['mae'], metrics['mae']),
                    'rmse': compute_degradation(baseline['rmse'], metrics['rmse']),
                    'r2_diff': metrics['r2'] - baseline['r2'],
                    'pearson_diff': metrics['pearson_corr'] - baseline['pearson_corr']
                }
            }

    save_path = os.path.join(output_dir, 'metrics_comparison.json')
    with open(save_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved comparison JSON to {save_path}")


def save_comparison_table(results, output_dir):
    """Save comparison table to text file."""
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, 'metrics_table.txt')

    with open(save_path, 'w') as f:
        # Redirect print to file
        import io
        old_stdout = sys.stdout
        sys.stdout = f
        print_comparison_table(results)
        sys.stdout = old_stdout

    print(f"Saved comparison table to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare Edge Ablation Results'
    )

    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing results subdirectories (normal, self_loops, zero_features)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for comparison (default: results-dir/comparison)'
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'comparison')

    print(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)

    if len(results) == 0:
        print("Error: No results found")
        sys.exit(1)

    # Print comparison table
    print_comparison_table(results)

    # Create visualizations
    print("\nGenerating comparison visualizations...")
    create_comparison_plots(results, args.output_dir)

    # Save comparison data
    save_comparison_json(results, args.output_dir)
    save_comparison_table(results, args.output_dir)

    print(f"\nComparison complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
