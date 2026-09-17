"""
models/nnunet.py — nnUNet segmentation model wrapper.

nnUNet handles its own preprocessing, so the workflow is different from UNet:
    1. prepare()  — converts numpy arrays to nnUNet folder structure
    2. fit()      — runs nnUNet training (long, use sbatch)
    3. load()     — loads trained checkpoint for inference
    4. predict()  — runs inference
    5. evaluate() — inherited from BaseSegmentor

Because training takes ~24 hours, fit() is meant to be run via sbatch.
Everything else (prepare, load, predict, evaluate) can run in a notebook.

Usage
-----
# Prepare and train (sbatch)
X_train, y_train, X_test, y_test = load_spleenex(test_fraction=0.2)
model = NNUNetSegmentor("Dataset101_SpleenUS", 101)
model.setup("/raid/mpsych/SPLEENUS/nnunet")
model.prepare(X_train, y_train, X_test, y_test)
model.fit()

# Load and evaluate (notebook)
model = NNUNetSegmentor("Dataset101_SpleenUS", 101)
model.setup("/raid/mpsych/SPLEENUS/nnunet")
model.load(folds=(0, 1, 2, 3, 4))
metrics = model.evaluate(X_test, y_test)
metrics = model.evaluate(X_robo, y_robo)  # unseen data
"""

import os
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from models.base import BaseSegmentor
from config import MODEL_ROOT

NNUNET_TRAINER  = "nnUNetTrainer"
NNUNET_PLANS    = "nnUNetPlans"
NNUNET_CONFIG   = "2d"
CHECKPOINT_NAME = "checkpoint_final.pth"


def _save_cases(images, masks, images_dir, labels_dir, file_format="png"):
    """
    Save image/mask pairs to nnUNet folder structure.

    Parameters
    ----------
    images     : np.ndarray  (N, H, W, 1)  float32
    masks      : np.ndarray  (N, H, W, 1)  float32
    images_dir : str  destination for images (imagesTr or imagesTs)
    labels_dir : str  destination for masks  (labelsTr or labelsTs), None for test
    file_format: str  "png" (default)
    """
    from PIL import Image

    os.makedirs(images_dir, exist_ok=True)
    if labels_dir is not None:
        os.makedirs(labels_dir, exist_ok=True)

    for i in range(len(images)):
        case_id = f"case_{i:04d}"

        img = (images[i].squeeze() * 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(
            os.path.join(images_dir, f"{case_id}_0000.{file_format}")
        )

        if labels_dir is not None:
            mask = masks[i].squeeze().astype(np.uint8)
            Image.fromarray(mask, mode="L").save(
                os.path.join(labels_dir, f"{case_id}.{file_format}")
            )


class NNUNetSegmentor(BaseSegmentor):
    """
    nnUNet segmentation model.

    Parameters
    ----------
    dataset_name : str  e.g. "Dataset101_SpleenUS"
    dataset_id   : int  e.g. 101
    threshold    : float  probability cutoff for binary mask (default 0.5)
    """

    def __init__(self, dataset_name, dataset_id, threshold=0.5):
        self.dataset_name = dataset_name
        self.dataset_id   = dataset_id
        self.threshold    = threshold
        self.base_dir     = None
        self._predictor   = None

    def setup(self, base_dir):
        """
        Set nnUNet environment variables and create required folders.

        Must be called before prepare(), fit(), or load().

        Parameters
        ----------
        base_dir : str  root folder for nnUNet data
        """
        self.base_dir = base_dir

        os.environ["nnUNet_raw"]          = os.path.join(base_dir, "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = os.path.join(base_dir, "nnUNet_preprocessed")
        os.environ["nnUNet_results"]      = os.path.join(base_dir, "nnUNet_results")

        for folder in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
            os.makedirs(os.path.join(base_dir, folder), exist_ok=True)

        print(f"nnUNet environment set up at: {base_dir}")

    def prepare(self, X_train, y_train, X_test, y_test, file_format="png"):
        """
        Convert numpy arrays to nnUNet folder structure and run preprocessing.

        Creates:
            imagesTr/ + labelsTr/  — training data
            imagesTs/ + labelsTs/  — test data

        Parameters
        ----------
        X_train     : np.ndarray  (N, H, W, 1)  float32  training images
        y_train     : np.ndarray  (N, H, W, 1)  float32  training masks
        X_test      : np.ndarray  (M, H, W, 1)  float32  test images
        y_test      : np.ndarray  (M, H, W, 1)  float32  test masks
        file_format : str         image format (default "png")
        """
        import json

        raw_dir = os.path.join(self.base_dir, "nnUNet_raw", self.dataset_name)

        # Training data
        images_tr_dir = os.path.join(raw_dir, "imagesTr")
        labels_tr_dir = os.path.join(raw_dir, "labelsTr")
        print(f"Saving {len(X_train)} training cases...")
        _save_cases(X_train, y_train, images_tr_dir, labels_tr_dir, file_format)

        # Test data
        images_ts_dir = os.path.join(raw_dir, "imagesTs")
        labels_ts_dir = os.path.join(raw_dir, "labelsTs")
        print(f"Saving {len(X_test)} test cases...")
        _save_cases(X_test, y_test, images_ts_dir, labels_ts_dir, file_format)

        # dataset.json
        dataset_json = {
            "channel_names": {"0": "grayscale"},
            "labels":        {"background": 0, "spleen": 1},
            "numTraining":   len(X_train),
            "file_ending":   f".{file_format}",
            "name":          self.dataset_name,
        }
        with open(os.path.join(raw_dir, "dataset.json"), "w") as f:
            json.dump(dataset_json, f, indent=2)

        print(f"Saved to {raw_dir}")
        print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

        # Run nnUNet preprocessing
        print("Running nnUNet preprocessing...")
        os.system(
            f"nnUNetv2_plan_and_preprocess -d {self.dataset_id} "
            f"--verify_dataset_integrity -c {NNUNET_CONFIG}"
        )
        print("Preprocessing done.")

    def fit(self, folds="all_cv"):
        """
        Run nnUNet training. This takes ~24 hours — use sbatch.

        Parameters
        ----------
        folds : str or int  "all_cv" for 5-fold CV, or a single fold number
        """
        if folds == "all_cv":
            for fold in range(5):
                print(f"\n=== Training fold {fold} ===")
                os.system(
                    f"nnUNetv2_train {self.dataset_id} {NNUNET_CONFIG} {fold}"
                )
        else:
            os.system(
                f"nnUNetv2_train {self.dataset_id} {NNUNET_CONFIG} {folds}"
            )

    def load(self, folds=(0, 1, 2, 3, 4), checkpoint=CHECKPOINT_NAME):
        """
        Load a trained nnUNet checkpoint for inference.

        Parameters
        ----------
        folds      : tuple  which folds to load (default all 5)
        checkpoint : str    checkpoint filename
        """
        model_folder = os.path.join(
            self.base_dir, "nnUNet_results", self.dataset_name,
            f"{NNUNET_TRAINER}__{NNUNET_PLANS}__{NNUNET_CONFIG}",
        )

        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False,
            allow_tqdm=False,
        )
        self._predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=folds,
            checkpoint_name=checkpoint,
        )
        print(f"nnUNet loaded from: {model_folder}")
        return self

    def predict(self, X):
        """
        Run inference and return binary masks.

        Works on any input size — nnUNet handles its own resizing internally.

        Parameters
        ----------
        X : np.ndarray  (N, H, W, 1)  float32

        Returns
        -------
        np.ndarray  (N, H, W, 1)  float32  values in {0, 1}
        """
        if self._predictor is None:
            raise RuntimeError("Call load() before predict().")

        results = []
        props   = {"spacing": [1, 1, 1]}

        for i in range(len(X)):
            img    = X[i].squeeze()                       # (H, W)
            img_nn = img[np.newaxis, np.newaxis, ...]     # (1, 1, H, W)

            predicted = self._predictor.predict_single_npy_array(
                img_nn, props, None, None,
                save_or_return_probabilities=True,
            )

            if isinstance(predicted, tuple):
                prob_map  = predicted[1]
                seg_probs = prob_map[1] if prob_map.shape[0] > 1 else prob_map[0]
            else:
                seg_probs = predicted.astype(np.float32)

            binary = (seg_probs > self.threshold).astype(np.float32)
            results.append(binary[..., np.newaxis])       # (H, W, 1)

        return np.array(results, dtype=np.float32)        # (N, H, W, 1)

    def save(self, path=None):
        """nnUNet manages its own checkpoints during training. This is a no-op."""
        print("nnUNet manages its own checkpoints. Use load() to load them.")