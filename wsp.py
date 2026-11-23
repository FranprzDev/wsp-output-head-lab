"""Reconstructed mathematical core; not an original benchmark release."""
import numpy as np


def fit_ridge(features, labels, regularization=100.0):
    """Fit on calibration data only. Return direction and scaling metadata."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
        raise ValueError("Expected N x D features and N labels, N >= 2")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Inputs must be finite")
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Regularization must be positive")
    mean_x = x.mean(axis=0)
    mean_y, scale_y = float(y.mean()), float(y.std())
    if scale_y == 0:
        raise ValueError("Labels must vary")
    centered = x - mean_x
    direction = np.linalg.solve(
        centered.T @ centered + regularization * np.eye(x.shape[1]),
        centered.T @ ((y - mean_y) / scale_y),
    )
    return direction, {"feature_mean": mean_x, "label_mean": mean_y,
                       "label_scale": scale_y}


def edit_rows(head, yes_id, no_id, direction, alpha=1.0):
    """Return a copy, changing exactly two rows. Input is a NumPy matrix."""
    head = np.asarray(head, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if head.ndim != 2 or direction.shape != (head.shape[1],):
        raise ValueError("Direction must match head width")
    if yes_id == no_id or not 0 <= yes_id < len(head) or not 0 <= no_id < len(head):
        raise ValueError("Expected two distinct valid row IDs")
    if not np.isfinite(head).all() or not np.isfinite(direction).all() or not np.isfinite(alpha):
        raise ValueError("Inputs must be finite")
    out = head.copy()
    out[yes_id] += alpha * direction / 2
    out[no_id] -= alpha * direction / 2
    return out


def pair_accuracy(scores, labels, gap=1.0):
    """Count ties as half. Pairs share clips; this is not an independence claim."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=float)
    if scores.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("Expected matching vectors")
    if gap <= 0 or not np.isfinite(gap) or not np.isfinite(scores).all() or not np.isfinite(labels).all():
        raise ValueError("Finite inputs and positive gap required")
    wins = 0.0
    count = 0
    for a in range(len(scores)):
        for b in range(a + 1, len(scores)):
            if abs(labels[a] - labels[b]) < gap:
                continue
            hi, lo = (a, b) if labels[a] > labels[b] else (b, a)
            wins += float(scores[hi] > scores[lo]) + .5 * float(scores[hi] == scores[lo])
            count += 1
    return {"pairs": count, "accuracy": wins / count if count else None}
