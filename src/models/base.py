"""
models/base.py — base class for all segmentation models.

Every model (UNet, nnUNet, MedSAM) inherits from BaseSegmentor and
implements the same four methods: fit, predict, evaluate, and visualize.

This means you can always do the same thing regardless of the model:

    model = UNetSegmentor("unet++")
    model.fit(X_train, y_train, epochs=50)
    metrics = model.evaluate(X_test, y_test)
    model.visualize(X_test, y_test, n=5)
"""

import numpy as np
from eval.metrics import evaluate as compute_metrics
from eval.metrics import visualize as show_predictions


class BaseSegmentor:
    """
    Base class for all segmentation models.

    Subclasses must implement:
        - fit(X_train, y_train, **kwargs)
        - predict(X) → np.ndarray of binary masks
    """

    def fit(self, X_train, y_train, **kwargs):
        """
        Train the model on images and masks.

        Parameters
        ----------
        X_train : np.ndarray  (N, H, W, 1)  float32
        y_train : np.ndarray  (N, H, W, 1)  float32
        """
        raise NotImplementedError("Subclasses must implement fit()")

    def predict(self, X):
        """
        Run inference and return binary masks.

        Parameters
        ----------
        X : np.ndarray  (N, H, W, 1)  float32

        Returns
        -------
        np.ndarray  (N, H, W, 1)  float32  values in {0, 1}
        """
        raise NotImplementedError("Subclasses must implement predict()")

    def evaluate(self, X, y):
        """
        Compute Dice and IoU on a test set.

        Parameters
        ----------
        X : np.ndarray  (N, H, W, 1)  float32  images
        y : np.ndarray  (N, H, W, 1)  float32  ground truth masks

        Returns
        -------
        dict with keys "dice" and "iou"
        """
        predictions = self.predict(X)
        return compute_metrics(predictions, y)

    def visualize(self, X, y, n=5, title=""):
        """
        Show n samples: image | ground truth | prediction | overlay.

        Parameters
        ----------
        X     : np.ndarray  (N, H, W, 1)  float32  images
        y     : np.ndarray  (N, H, W, 1)  float32  ground truth masks
        n     : int         number of samples to show (default 5)
        title : str         optional figure title
        """
        predictions = self.predict(X[:n])
        show_predictions(X[:n], y[:n], predictions, n=n, title=title)