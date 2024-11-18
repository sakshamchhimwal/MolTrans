import numpy as np
from typing import List, Tuple, Dict

def analyze_loss(loss_history: List[float]) -> Dict:
    final_loss = loss_history[-1]
    avg_loss = np.mean(loss_history)
    min_loss = np.min(loss_history)

    stats = {
        'final_loss': final_loss,
        'average_loss': avg_loss,
        'minimum_loss': min_loss,
        'total_iterations': len(loss_history)
    }

    return stats


def get_model_loss_summary(loss_history):
    stats = analyze_loss(loss_history)
    summary = (
        f"\n{'=' * 50}\n"
        f"Model Training Summary:\n"
        f"{'-' * 50}\n"
        f"{'Final Loss:':<20} {stats['final_loss']:.4f}\n"
        f"{'Average Loss:':<20} {stats['average_loss']:.4f}\n"
        f"{'Minimum Loss:':<20} {stats['minimum_loss']:.4f}\n"
        f"{'Total Iterations:':<20} {stats['total_iterations']:.4f}\n"
        f"{'=' * 50}"
    )
    return summary