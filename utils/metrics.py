import numpy as np


def confusion_counts(prediction, target, threshold=0.5):
    pred = np.asarray(prediction) >= threshold
    truth = np.asarray(target) >= 0.5
    return {
        "tp": int(np.logical_and(pred, truth).sum()),
        "tn": int(np.logical_and(~pred, ~truth).sum()),
        "fp": int(np.logical_and(pred, ~truth).sum()),
        "fn": int(np.logical_and(~pred, truth).sum()),
    }


def metrics_from_counts(counts):
    tp, tn, fp, fn = (counts[k] for k in ("tp", "tn", "fp", "fn"))
    safe = lambda num, den: float(num / den) if den else 0.0
    return {
        "DSC": safe(2 * tp, 2 * tp + fp + fn),
        "IoU": safe(tp, tp + fp + fn),
        "SE": safe(tp, tp + fn),
        "SP": safe(tn, tn + fp),
        "ACC": safe(tp + tn, tp + tn + fp + fn),
        "Precision": safe(tp, tp + fp),
    }


def add_counts(total, update):
    for key in total:
        total[key] += update[key]
    return total
