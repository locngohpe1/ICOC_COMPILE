# visualize_training.py - Create publication-quality training plots
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_training():
    """
    Create paper-quality training visualizations
    """
    history_path = 'models/training_history.csv'

    if not os.path.exists(history_path):
        print("❌ Training history not found!")
        print("   Expected: models/training_history.csv")
        return

    print("=" * 70)
    print("VISUALIZING TRAINING HISTORY - V1 (91.5%)")
    print("=" * 70)

    # Load history
    df = pd.read_csv(history_path)

    print(f"\n📊 Loaded training data:")
    print(f"   Total epochs: {len(df)}")
    print(f"   Best val accuracy: {df['val_accuracy'].max():.4f} (epoch {df['val_accuracy'].idxmax() + 1})")
    print(f"   Final train accuracy: {df['train_accuracy'].iloc[-1]:.4f}")
    print(f"   Final val accuracy: {df['val_accuracy'].iloc[-1]:.4f}")

    if not os.path.exists('results'):
        os.makedirs('results')

    # ==================== PLOT 1: TRAINING CURVES (MAIN) ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Loss curves
    axes[0, 0].plot(df['epoch'], df['train_loss'], 'o-', linewidth=2, markersize=4,
                    color='#3498db', label='Training Loss')
    axes[0, 0].plot(df['epoch'], df['val_loss'], 's-', linewidth=2, markersize=4,
                    color='#e74c3c', label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold', pad=15)
    axes[0, 0].legend(fontsize=11, loc='upper right')
    axes[0, 0].grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = df['val_accuracy'].idxmax() + 1
    best_val_loss = df['val_loss'].iloc[df['val_accuracy'].idxmax()]
    axes[0, 0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, linewidth=2)
    axes[0, 0].text(best_epoch, best_val_loss, f' Best (Epoch {best_epoch})',
                    fontsize=10, color='green', fontweight='bold')

    # Subplot 2: Accuracy curves
    axes[0, 1].plot(df['epoch'], df['train_accuracy'] * 100, 'o-', linewidth=2, markersize=4,
                    color='#3498db', label='Training Accuracy')
    axes[0, 1].plot(df['epoch'], df['val_accuracy'] * 100, 's-', linewidth=2, markersize=4,
                    color='#e74c3c', label='Validation Accuracy')
    axes[0, 1].axhline(y=df['val_accuracy'].max() * 100, color='green', linestyle='--',
                       linewidth=2, alpha=0.7, label=f'Best Val: {df["val_accuracy"].max() * 100:.2f}%')
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold', pad=15)
    axes[0, 1].legend(fontsize=11, loc='lower right')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([50, 100])

    # Mark best epoch
    axes[0, 1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5, linewidth=2)

    # Subplot 3: Overfitting gap
    overfitting_gap = (df['train_accuracy'] - df['val_accuracy']) * 100
    axes[1, 0].plot(df['epoch'], overfitting_gap, 'o-', color='#e67e22', linewidth=2, markersize=4)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    axes[1, 0].fill_between(df['epoch'], 0, overfitting_gap, alpha=0.3, color='#e67e22')
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Gap (%)', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Overfitting Gap (Train - Val Accuracy)', fontsize=14, fontweight='bold', pad=15)
    axes[1, 0].grid(True, alpha=0.3)

    # Add annotation for final gap
    final_gap = overfitting_gap.iloc[-1]
    axes[1, 0].text(len(df) - 5, final_gap, f'Final: {final_gap:.2f}%',
                    fontsize=10, fontweight='bold', color='#e67e22')

    # Subplot 4: Learning rate schedule
    axes[1, 1].plot(df['epoch'], df['learning_rate'], 'o-', color='#9b59b6',
                    linewidth=2, markersize=5)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold', pad=15)
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3, which='both')

    # Add LR reduction annotations
    lr_changes = df[df['learning_rate'].diff() != 0]
    for idx, row in lr_changes.iterrows():
        if idx > 0:  # Skip first epoch
            axes[1, 1].axvline(x=row['epoch'], color='red', linestyle='--', alpha=0.4, linewidth=1.5)
            axes[1, 1].text(row['epoch'], row['learning_rate'], f" LR: {row['learning_rate']:.6f}",
                            fontsize=9, color='red')

    plt.suptitle('GoogLeNet Training History - Obstacle Classification',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('results/training_curves_complete.png', dpi=300, bbox_inches='tight')
    print("\n   ✅ Saved: results/training_curves_complete.png")

    # ==================== PLOT 2: ACCURACY ONLY (FOR PAPER) ====================
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.plot(df['epoch'], df['train_accuracy'] * 100, 'o-', linewidth=2.5, markersize=5,
            color='#3498db', label='Training', alpha=0.8)
    ax.plot(df['epoch'], df['val_accuracy'] * 100, 's-', linewidth=2.5, markersize=5,
            color='#e74c3c', label='Validation', alpha=0.8)
    ax.axhline(y=df['val_accuracy'].max() * 100, color='green', linestyle='--',
               linewidth=2, alpha=0.6, label=f'Best Validation: {df["val_accuracy"].max() * 100:.2f}%')

    # Mark best epoch
    best_idx = df['val_accuracy'].idxmax()
    ax.scatter(df['epoch'].iloc[best_idx], df['val_accuracy'].iloc[best_idx] * 100,
               s=200, color='green', marker='*', zorder=5, edgecolors='black', linewidth=2)

    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Training Progress - GoogLeNet Obstacle Classifier',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([70, 100])

    plt.tight_layout()
    plt.savefig('results/accuracy_curve_paper.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/accuracy_curve_paper.png")

    # ==================== PLOT 3: CONVERGENCE ANALYSIS ====================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: First 15 epochs (rapid learning phase)
    early_df = df[df['epoch'] <= 15]
    axes[0].plot(early_df['epoch'], early_df['val_accuracy'] * 100, 'o-',
                 linewidth=2.5, markersize=6, color='#e74c3c')
    axes[0].scatter(best_epoch, df['val_accuracy'].iloc[best_idx] * 100,
                    s=200, color='green', marker='*', zorder=5, edgecolors='black', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Early Training Phase (Epochs 1-15)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([80, 95])

    # Right: Plateau phase
    late_df = df[df['epoch'] > 15]
    if len(late_df) > 0:
        axes[1].plot(late_df['epoch'], late_df['val_accuracy'] * 100, 'o-',
                     linewidth=2.5, markersize=6, color='#3498db')
        axes[1].axhline(y=df['val_accuracy'].max() * 100, color='green',
                        linestyle='--', linewidth=2, alpha=0.6)
        axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Plateau Phase (Epochs 16-{len(df)})', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([85, 93])

    plt.tight_layout()
    plt.savefig('results/convergence_analysis.png', dpi=300, bbox_inches='tight')
    print("   ✅ Saved: results/convergence_analysis.png")

    # ==================== STATISTICS TABLE ====================
    print("\n" + "=" * 70)
    print("📊 TRAINING STATISTICS")
    print("=" * 70)

    print(f"\nConvergence:")
    print(f"  Epoch to best val accuracy: {best_epoch}")
    print(f"  Best validation accuracy:   {df['val_accuracy'].max() * 100:.2f}%")
    print(f"  Final training accuracy:    {df['train_accuracy'].iloc[-1] * 100:.2f}%")
    print(f"  Final validation accuracy:  {df['val_accuracy'].iloc[-1] * 100:.2f}%")

    print(f"\nLearning Rate:")
    print(f"  Initial LR:  {df['learning_rate'].iloc[0]:.6f}")
    print(f"  Final LR:    {df['learning_rate'].iloc[-1]:.6f}")
    print(f"  LR reductions: {len(df[df['learning_rate'].diff() != 0]) - 1}")

    print(f"\nOverfitting:")
    print(
        f"  Overfitting at best epoch:  {(df['train_accuracy'].iloc[best_idx] - df['val_accuracy'].iloc[best_idx]) * 100:.2f}%")
    print(f"  Overfitting at final epoch: {overfitting_gap.iloc[-1]:.2f}%")

    print(f"\nLoss:")
    print(f"  Final train loss: {df['train_loss'].iloc[-1]:.4f}")
    print(f"  Final val loss:   {df['val_loss'].iloc[-1]:.4f}")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 70)
    print("✅ VISUALIZATION COMPLETED!")
    print("=" * 70)
    print(f"\nGenerated Files (High Resolution - 300 DPI):")
    print(f"  📊 results/training_curves_complete.png    (4-panel overview)")
    print(f"  📊 results/accuracy_curve_paper.png        (clean for paper)")
    print(f"  📊 results/convergence_analysis.png        (early vs late)")

    print("\n" + "=" * 70)
    print("🎯 All plots ready for Q1 paper!")
    print("=" * 70)


if __name__ == "__main__":
    visualize_training()