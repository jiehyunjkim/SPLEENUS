"""
models/unet.py — UNet family segmentation models.

Supports: "unet", "unet++", "unet3++"
Uses keras_unet_collection under the hood.

Usage:
    from models.unet import UNetSegmentor

    model = UNetSegmentor("unet++")
    model.fit(X_train, y_train, epochs=50)
    metrics = model.evaluate(X_test, y_test)
    model.visualize(X_test, y_test, n=5)
    model.save("/raid/mpsych/SPLEENUS/models/unetpp.keras")
"""

import numpy as np
import tensorflow as tf
from keras_unet_collection import models as kuc_models

from models.base import BaseSegmentor
from config import IMAGE_SIZE, MODEL_ROOT

# Model architecture settings
FILTER_NUM       = [64, 128, 256, 512]
BATCH_SIZE       = 16
LEARNING_RATE    = 1e-4
VALIDATION_SPLIT = 0.1

SUPPORTED_MODELS = ["unet", "unet++", "unet3++"]

# =============================================================================
# Loss and metrics
# =============================================================================

# Loss — TF library (cleaner and more reliable than custom implementation)
_bce_loss  = tf.keras.losses.BinaryCrossentropy()
_dice_loss = tf.keras.losses.Dice()

@tf.keras.utils.register_keras_serializable()
def bce_dice_loss(y_true, y_pred):
    """Binary cross-entropy + Dice loss for training."""
    return _bce_loss(y_true, y_pred) + _dice_loss(y_true, y_pred)


# Dice and IoU metrics
@tf.keras.utils.register_keras_serializable()
def dice_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

_iou_metric  = tf.keras.metrics.BinaryIoU(name="iou")


# =============================================================================
# Model factory
# =============================================================================

def _build_model(name):
    """Build a Keras model by name."""
    input_size = (IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    if name == "unet":
        return kuc_models.unet_2d(
            input_size=input_size,
            filter_num=FILTER_NUM,
            n_labels=1,
            output_activation="Sigmoid",
        )
    elif name == "unet++":
        return kuc_models.unet_plus_2d(
            input_size=input_size,
            filter_num=FILTER_NUM,
            n_labels=1,
            output_activation="Sigmoid",
        )
    elif name == "unet3++":
        return kuc_models.unet_3plus_2d(
            input_size=input_size,
            filter_num_down=FILTER_NUM,
            n_labels=1,
            output_activation="Sigmoid",
        )
    else:
        raise ValueError(f"Unknown model '{name}'. Choose from: {SUPPORTED_MODELS}")


# =============================================================================
# UNetSegmentor
# =============================================================================

class UNetSegmentor(BaseSegmentor):
    """
    Segmentation model using the UNet family.

    Parameters
    ----------
    model_name : str    one of "unet", "unet++", "unet3++"
    threshold  : float  probability cutoff for binary mask (default 0.5)
    """

    def __init__(self, model_name, threshold=0.5):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {SUPPORTED_MODELS}")

        self.model_name = model_name
        self.threshold  = threshold
        self.model      = _build_model(model_name)

    def fit(self, X_train, y_train, epochs=50, callbacks=None):
        """
        Train the model.

        Parameters
        ----------
        X_train   : np.ndarray  (N, H, W, 1)  float32
        y_train   : np.ndarray  (N, H, W, 1)  float32
        epochs    : int         number of training epochs (default 50)
        callbacks : list        Keras callbacks (optional)
                                If None, uses EarlyStopping + ModelCheckpoint
        """
        safe_name = self.model_name.replace("++", "pp").replace("+", "p")
        save_path = f"{MODEL_ROOT}/best_{safe_name}.keras"

        if callbacks is None:
            callbacks = [
                # tf.keras.callbacks.EarlyStopping(
                #     monitor="val_dice_metric",
                #     patience=20,
                #     mode="max",
                #     restore_best_weights=True,
                #     verbose=1,
                # ),
                tf.keras.callbacks.ModelCheckpoint(
                    save_path,
                    monitor="val_dice_metic",
                    save_best_only=True,
                    mode="max",
                    verbose=1,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1,
                ),
            ]

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=bce_dice_loss,
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                dice_metric,
                _iou_metric,
            ],
        )

        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=BATCH_SIZE,
            validation_split=VALIDATION_SPLIT,
            callbacks=callbacks,
            verbose=1,
        )
        return self

    def predict(self, X):
        """
        Run inference and return binary masks.

        Applies threshold (default 0.5) to convert probabilities to 0/1.

        Parameters
        ----------
        X : np.ndarray  (N, H, W, 1)  float32

        Returns
        -------
        np.ndarray  (N, H, W, 1)  float32  values in {0, 1}
        """
        probabilities = self.model.predict(X, verbose=0)
        return (probabilities > self.threshold).astype(np.float32)

    def save(self, path=None):
        """
        Save the model.

        Parameters
        ----------
        path : str  save path (optional)
               If None, saves to MODEL_ROOT/best_{model_name}.keras
        """
        if path is None:
            safe_name = self.model_name.replace("++", "pp").replace("+", "p")
            path = f"{MODEL_ROOT}/best_{safe_name}.keras"

        self.model.save(path)
        print(f"Saved to {path}")

    def load(self, path):
        """
        Load a saved model.

        Parameters
        ----------
        path : str  path to saved .keras file
        """
        self.model = tf.keras.models.load_model(
            path,
            custom_objects={
                "bce_dice_loss": bce_dice_loss,
                "dice_metric":   dice_metric,
            },
        )
        print(f"Loaded from {path}")
        return self