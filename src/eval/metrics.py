"""
eval/metrics.py — all evaluation functions in one place.

All models use these functions via BaseSegmentor.evaluate().

Available functions:
    dice_score(pred, gt)        → float
    iou_score(pred, gt)         → float
    evaluate(pred, gt)          → dict {"dice": float, "iou": float}
    visualize(X, y, preds, n)   → matplotlib figure
"""

import numpy as np
import matplotlib.pyplot as plt

SMOOTH = 1e-6  # prevents division by zero


def dice_score(pred, gt):
    """
    Compute Dice score between predicted and ground truth masks.
    """
    pred = pred.flatten().astype(np.float32)
    gt   = gt.flatten().astype(np.float32)

    intersection = np.sum(pred * gt)
    return (2.0 * intersection + SMOOTH) / (np.sum(pred) + np.sum(gt) + SMOOTH)


def iou_score(pred, gt):
    """
    Compute IoU (Intersection over Union) between predicted and ground truth masks.
    """
    pred = pred.flatten().astype(np.float32)
    gt   = gt.flatten().astype(np.float32)

    intersection = np.sum(pred * gt)
    union = np.sum(pred) + np.sum(gt) - intersection
    return (intersection + SMOOTH) / (union + SMOOTH)


def evaluate(predictions, ground_truth):
    """
    Compute all metrics for a batch of predictions.
    """
    dice = dice_score(predictions, ground_truth)
    iou  = iou_score(predictions, ground_truth)

    results = {
        "dice": float(dice),
        "iou":  float(iou),
    }

    print(f"Dice: {dice:.4f} | IoU: {iou:.4f}")
    return results


def visualize(X, y, predictions, n=5, title=""):
    """
    Show n samples side by side: image | ground truth | prediction | overlay.

    Parameters
    ----------
    X           : np.ndarray  (N, H, W, 1)  float32  images
    y           : np.ndarray  (N, H, W, 1)  float32  ground truth masks
    predictions : np.ndarray  (N, H, W, 1)  float32  predicted masks
    n           : int         number of samples to show (default 5)
    title       : str         optional title for the whole figure
    """
    n = min(n, len(X))

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img  = X[i].squeeze()
        gt   = y[i].squeeze()
        pred = predictions[i].squeeze()

        dice = dice_score(pred, gt)

        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(f"Image #{i}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title("Ground truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap="gray")
        axes[i, 2].set_title(f"Prediction  Dice {dice:.3f}")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(img, cmap="gray")
        axes[i, 3].imshow(np.ma.masked_where(pred == 0, pred), cmap="autumn", alpha=0.5)
        axes[i, 3].set_title("Overlay")
        axes[i, 3].axis("off")

    if title:
        plt.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()