"""
models/sam_models.py — SAM3 and MedSAM3 segmentation models.

Both models are SAM3-family, with one real difference:
    - SAM3    : baseline segment model. Accept text and trained model for bbox.
    - MedSAM3 : applied medical-domain LoRA. Baseline accept only text.

Usage
-----
    from models.sam_models import SAM3Segmentor, MedSAM3Segmentor
    from models.nnunet import NNUNetSegmentor

    nn_model = NNUNetSegmentor(...); nn_model.load(...)

    sam = SAM3Segmentor().load()
    metrics = sam.evaluate_with_bbox(X_test, y_test, bbox_model=nn_model)

    medsam = MedSAM3Segmentor().load()
    metrics = medsam.evaluate(X_test, y_test)                                 # text-only
    metrics = medsam.evaluate_with_bbox(X_test, y_test, bbox_model=nn_model)  # text + bbox
"""
import os
from typing import Optional

import numpy as np
from PIL import Image as PILImage

from config import RUNS_ROOT
from models.base import BaseSegmentor
from models.sam.inference import build_medsam3_predictor, build_sam3_predictor
from models.sam.train import train_medsam3, train_sam3

TEXT_PROMPT = "spleen"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_pil_rgb(image_array: np.ndarray) -> PILImage.Image:
    """(H, W, 1) float32 in [0,1] -> PIL RGB. SAM3/MedSAM3 expect RGB input."""
    gray = (image_array.squeeze() * 255).astype(np.uint8)
    return PILImage.fromarray(gray).convert("RGB")


def _to_uint8_rgb(images: np.ndarray) -> np.ndarray:
    """(N, H, W, 1) float32 in [0,1] -> (N, H, W, 3) uint8, for training."""
    if images.shape[-1] == 1:
        return np.stack([(images.squeeze(axis=-1) * 255).astype(np.uint8)] * 3, axis=-1)
    return (images * 255).astype(np.uint8)


def _mask_to_bbox(mask_2d: np.ndarray) -> Optional[list]:
    """[x_min, y_min, x_max, y_max] from a 2D binary mask, or None if empty."""
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def _boxes_from_model(bbox_model, X: np.ndarray) -> list:
    """Batch-predict masks with bbox_model ONCE, then reduce each to a box.

    Not one image at a time -- nnUNet in particular is much cheaper to run
    as a single batched forward pass than called per image. A `None` entry
    means the detector found nothing for that image; the predictors below
    just treat box=None as "no box for this one" rather than erroring.
    """
    masks = bbox_model.predict(X)  # (N, H, W, 1)
    return [_mask_to_bbox(masks[i].squeeze()) for i in range(len(X))]


def _run_predictor(predict_fn, X: np.ndarray, boxes: Optional[list], threshold: float) -> np.ndarray:
    out = np.zeros((len(X), X.shape[1], X.shape[2], 1), dtype=np.float32)
    for i in range(len(X)):
        box = boxes[i] if boxes is not None else None
        result = predict_fn(_to_pil_rgb(X[i]), TEXT_PROMPT, box=box, threshold=threshold)
        #if result["masks"] is not None and result["scores"] is not None:
        #    best_idx = int(np.argmax(result["scores"]))
        #    out[i, ..., 0] = result["masks"][best_idx].astype(np.float32)
        if result["masks"] is not None:
            out[i, ..., 0] = result["masks"][0].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# SAM3Segmentor
# ---------------------------------------------------------------------------

class SAM3Segmentor(BaseSegmentor):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._predict_fn = None
        self.weights_path = None

    def load(self, weights_path: Optional[str] = None) -> "SAM3Segmentor":
        """weights_path: fine-tuned .safetensors checkpoint, or None for
        the public base SAM3 weights."""
        self._predict_fn = build_sam3_predictor(weights_path=weights_path)
        self.weights_path = weights_path
        return self

    def fit(
        self, X_train, y_train, epochs: int = 10,
        output_dir: Optional[str] = None, box_source: str = "gt",
    ) -> "SAM3Segmentor":
        """box_source="gt" (default) trains with a box derived from each ground-truth mask.
        evaluate_with_bbox() sources its boxes from a models (UNet/nnUNet). 
        Pass box_source="none" to train text-only instead.
        """
        if output_dir is None:
            output_dir = os.path.join(RUNS_ROOT, "sam3_finetune")
        images = _to_uint8_rgb(X_train)
        masks = y_train.squeeze(axis=-1).astype(bool)
        self.weights_path = train_sam3(
            images, masks, output_dir=output_dir, epochs=epochs,
            box_source=box_source, text_prompt=TEXT_PROMPT,
        )
        print(f"[SAM3] fine-tuned weights -> {self.weights_path}")
        return self

    def predict(self, X) -> np.ndarray:
        if self._predict_fn is None:
            raise RuntimeError("Call load() before predict().")
        return _run_predictor(self._predict_fn, X, boxes=None, threshold=self.threshold)

    def predict_with_bbox(self, X, bbox_model) -> np.ndarray:
        """bbox_model: anything exposing .predict(X) -> (N,H,W,1) masks
        (NNUNetSegmentor / UNetSegmentor)."""
        if self._predict_fn is None:
            raise RuntimeError("Call load() before predict_with_bbox().")
        boxes = _boxes_from_model(bbox_model, X)
        return _run_predictor(self._predict_fn, X, boxes=boxes, threshold=self.threshold)

    def evaluate(self, X, y) -> dict:
        from eval.metrics import evaluate
        return evaluate(self.predict(X), y)

    def evaluate_with_bbox(self, X, y, bbox_model) -> dict:
        from eval.metrics import evaluate
        return evaluate(self.predict_with_bbox(X, bbox_model), y)

    def visualize_with_bbox(self, X, y, bbox_model, n: int = 5) -> None:
        from eval.metrics import visualize
        visualize(X[:n], y[:n], self.predict_with_bbox(X[:n], bbox_model), n=n, title="SAM3 + bbox")


# ---------------------------------------------------------------------------
# MedSAM3Segmentor
# ---------------------------------------------------------------------------

class MedSAM3Segmentor(BaseSegmentor):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self._predict_fn = None
        self.weights_path = None

    def load(self, weights_path: Optional[str] = None) -> "MedSAM3Segmentor":
        """weights_path: fine-tuned LoRA .pt checkpoint, or None to
        download the public lal-Joey/MedSAM3_v1 weights."""
        self._predict_fn = build_medsam3_predictor(lora_weights_path=weights_path)
        self.weights_path = weights_path
        return self

    def fit(self, X_train, y_train, epochs: int = 10, output_dir: Optional[str] = None, box_source="none") -> "MedSAM3Segmentor":
        if output_dir is None:
            output_dir = os.path.join(RUNS_ROOT, "medsam3_finetune")
        images = _to_uint8_rgb(X_train)
        masks = y_train.squeeze(axis=-1).astype(bool)
        self.weights_path = train_medsam3(images, masks, output_dir=output_dir, category=TEXT_PROMPT, epochs=epochs, box_source=box_source)
        print(f"[MedSAM3] fine-tuned weights -> {self.weights_path}")
        return self

    def predict(self, X) -> np.ndarray:
        """Text-only."""
        if self._predict_fn is None:
            raise RuntimeError("Call load() before predict().")
        return _run_predictor(self._predict_fn, X, boxes=None, threshold=self.threshold)

    def predict_with_bbox(self, X, bbox_model) -> np.ndarray:
        """bbox_model: anything exposing .predict(X) -> (N,H,W,1) masks
        (NNUNetSegmentor / UNetSegmentor). Text + box """
        if self._predict_fn is None:
            raise RuntimeError("Call load() before predict_with_bbox().")
        boxes = _boxes_from_model(bbox_model, X)
        return _run_predictor(self._predict_fn, X, boxes=boxes, threshold=self.threshold)

    def evaluate(self, X, y) -> dict:
        from eval.metrics import evaluate
        return evaluate(self.predict(X), y)

    def evaluate_with_bbox(self, X, y, bbox_model) -> dict:
        from eval.metrics import evaluate
        return evaluate(self.predict_with_bbox(X, bbox_model), y)

    def visualize_with_bbox(self, X, y, bbox_model, n: int = 5) -> None:
        from eval.metrics import visualize
        visualize(X[:n], y[:n], self.predict_with_bbox(X[:n], bbox_model), n=n, title="MedSAM3 + bbox")


        
