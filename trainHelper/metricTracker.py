import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime


def display_validation_metrics(epoch: int, auc: float, auprc: float, f1: float, loss: float):
    print("\n" + "=" * 50)
    print(f"Validation Results - Epoch {epoch}")
    print("-" * 50)
    print(f"{'AUROC:':<15} {auc:.4f}")
    print(f"{'AUPRC:':<15} {auprc:.4f}")
    print(f"{'F1 Score:':<15} {f1:.4f}")
    print(f"{'Loss:':<15} {loss:.4f}")
    print(f"{'Time:':<15} {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50 + "\n")

def display_testing_metrics(auc: float, auprc: float, f1: float, loss: float):
    print("\n" + "=" * 50)
    print(f"Testing Results")
    print("-" * 50)
    print(f"{'AUROC:':<15} {auc:.4f}")
    print(f"{'AUPRC:':<15} {auprc:.4f}")
    print(f"{'F1 Score:':<15} {f1:.4f}")
    print(f"{'Loss:':<15} {loss:.4f}")
    print(f"{'Time:':<15} {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 50 + "\n")

def update_metrics_history(
        metrics_history: List[Dict],
        epoch: int,
        auc: float,
        auprc: float,
        f1: float,
        loss: float
) -> Tuple[List[Dict], bool]:
    current_metrics = {
        'epoch': epoch,
        'auc': auc,
        'auprc': auprc,
        'f1': f1,
        'loss': loss,
        'timestamp': datetime.now().strftime('%H:%M:%S')
    }

    metrics_history.append(current_metrics)

    is_best = auc >= max([m['auc'] for m in metrics_history]) if metrics_history else True

    return metrics_history, is_best


def get_metrics_summary(metrics_history: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(metrics_history)


def display_metrics_summary(df_summary: pd.DataFrame):
    formatted_df = df_summary.copy()
    formatted_df = formatted_df.round({
        'auc': 4,
        'auprc': 4,
        'f1': 4,
        'loss': 4
    })

    # 1. Full table with all metrics
    print("\n=== COMPLETE TRAINING METRICS SUMMARY ===")
    print(formatted_df.to_string(index=False))

    # 2. Best performance for each metric
    print("\n=== BEST PERFORMANCE METRICS ===")
    best_metrics = {
        'Best AUROC': (formatted_df['auc'].max(), formatted_df.loc[formatted_df['auc'].idxmax(), 'epoch']),
        'Best AUPRC': (formatted_df['auprc'].max(), formatted_df.loc[formatted_df['auprc'].idxmax(), 'epoch']),
        'Best F1': (formatted_df['f1'].max(), formatted_df.loc[formatted_df['f1'].idxmax(), 'epoch']),
        'Lowest Loss': (formatted_df['loss'].min(), formatted_df.loc[formatted_df['loss'].idxmin(), 'epoch'])
    }

    for metric, (value, epoch) in best_metrics.items():
        print(f"{metric:<15} {value:.4f} (Epoch {epoch})")



