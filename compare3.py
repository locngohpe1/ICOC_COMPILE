# compare_versions.py - Compare V1, V2, V3 for ablation study
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def compare_versions():
    """
    Create comparison plots for ablation study
    """
    print("=" * 70)
    print("COMPARING V1, V2, V3 - ABLATION STUDY")
    print("=" * 70)

    # Load all histories
    versions = {}
    for v in ['', '_v2', '_v3']:
        path = f'models/training_history{v}.csv'
        if os.path.exists(path):
            versions[v if v else '_v1'] = pd.read_csv(path)
            print(f"   ✅ Loaded: {path}")
        else:
            print(f"   ⚠️  Not found: {path}")

    if len(versions) < 2:
        print("\n❌ Need at least 2 versions to compare!")
        return

    if not os.path.exists('results'):
        os.makedirs('results')

    # ==================== COMPARISON PLOT ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = {'_v1': '#3498db', '_v2': '#e74c3c', '_v3': '#2ecc71'}
    labels = {'_v1': 'V1 (Baseline)', '_v2': 'V2 (Strong Reg)', '_v3': 'V3 (Moderate Reg)'}

    # Subplot 1: Validation accuracy
    for name, df in versions.items():
        axes[0, 0].plot(df['epoch'], df['val_accuracy'] * 100, 'o-',
                        linewidth=2, markersize=4, color=colors[name],
                        label=f"{labels[name]}: {df['val_accuracy'].max() * 100:.2f}%", alpha=0.8)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Training accuracy
    for name, df in versions.items():
        axes[0, 1].plot(df['epoch'], df['train_accuracy'] * 100, 'o-',
                        linewidth=2, markersize=4, color=colors[name],
                        label=labels[name], alpha=0.8)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Overfitting gap
    for name, df in versions.items():
        gap = (df['train_accuracy'] - df['val_accuracy']) * 100
        axes[1, 0].plot(df['epoch'], gap, 'o-', linewidth=2, markersize=4,
                        color=colors[name], label=f"{labels[name]}: {gap.iloc[-1]:.2f}%", alpha=0.8)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Gap (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Overfitting Gap Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Summary bar chart
    summary_data = {}
    for name, df in versions.items():
        best_val = df['val_accuracy'].max() * 100
        final_train = df['train_accuracy'].iloc[-1] * 100
        gap = (df['train_accuracy'].iloc[-1] - df['val_accuracy'].iloc[-1]) * 100
        summary_data[labels[name]] = [best_val, gap]

    x = np.arange(len(summary_data))
    width = 0.35

    vals = [v[0] for v in summary_data.values()]
    gaps = [v[1] for v in summary_data.values()]

    bars1 = axes[1, 1].bar(x - width / 2, vals, width, label='Best Val Acc (%)',
                           color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = axes[1, 1].bar(x + width / 2, gaps, width, label='Overfitting Gap (%)',
                           color='#e74c3c', alpha=0.8, edgecolor='black')

    # Add values on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[1, 1].set_ylabel('Value (%)', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Summary Comparison', fontsize=14, fontweight='bold', pad=15)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(summary_data.keys(), rotation=15, ha='right')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Ablation Study: Regularization Techniques',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/version_comparison.png', dpi=300, bbox_inches='tight')
    print("\n   ✅ Saved: results/version_comparison.png")

    # Print summary table
    print("\n" + "=" * 70)
    print("📊 SUMMARY TABLE")
    print("=" * 70)
    print(f"\n{'Version':<20} {'Best Val Acc':<15} {'Final Train':<15} {'Gap':<10} {'Epochs':<10}")
    print("-" * 70)
    for name, df in versions.items():
        best_val = df['val_accuracy'].max() * 100
        final_train = df['train_accuracy'].iloc[-1] * 100
        gap = (df['train_accuracy'].iloc[-1] - df['val_accuracy'].iloc[-1]) * 100
        epochs = len(df)
        print(f"{labels[name]:<20} {best_val:>13.2f}%  {final_train:>13.2f}%  {gap:>8.2f}%  {epochs:>8d}")

    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETED!")
    print("=" * 70)


if __name__ == "__main__":
    compare_versions()