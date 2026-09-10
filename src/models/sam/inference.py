"""Prediction for SAM3 (HuggingFace) and MedSAM3 (native sam3 + LoRA)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from .lora import LoRAConfig, apply_lora_to_model, load_lora_weights

PathLike = Union[str, Path]
PredictFn = Callable[..., dict]

_HF_MODEL_ID = "facebook/sam3"
_DEFAULT_MEDSAM3_LORA_REPO = "lal-Joey/MedSAM3_v1"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_pil(image) -> PILImage.Image:
    if isinstance(image, (str, Path)):
        return PILImage.open(image).convert("RGB")
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    arr = np.squeeze(np.asarray(image))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = (arr * 255 if arr.max() <= 1.0 else arr).astype(np.uint8)
    return PILImage.fromarray(arr, mode="RGB")


def _postprocess_mask(mask_2d: np.ndarray) -> np.ndarray:
    """Morphological close + keep only the largest connected component.
       closes small holes from thresholding noise, then drops any disconnected speckle predictions.
    """
    m = mask_2d.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    return (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(bool)


def _empty_result(pil_image: PILImage.Image) -> dict:
    return {"boxes": None, "scores": None, "masks": None, "num_detections": 0, "image": pil_image}


# ---------------------------------------------------------------------------
# SAM3 (HuggingFace facebook/sam3)
# ---------------------------------------------------------------------------

def build_sam3_predictor(
    bbox_fn: Optional[Callable[[PILImage.Image], Optional[list]]] = None,
    weights_path: Optional[PathLike] = None,
    device: Optional[torch.device] = None,
    postprocess: bool = True,
) -> PredictFn:
    from transformers import Sam3Model, Sam3Processor

    device = device or _pick_device()
    model = Sam3Model.from_pretrained(_HF_MODEL_ID).to(device)
    processor = Sam3Processor.from_pretrained(_HF_MODEL_ID)

    if weights_path is not None:
        from safetensors.torch import load_file

        missing, unexpected = model.load_state_dict(load_file(str(weights_path)), strict=False)
        print(f"[sam3] loaded fine-tuned weights from {weights_path} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()
    print(f"[sam3] ready on {device}")

    @torch.no_grad()
    def predict(image, text_prompt: str, box: Optional[list] = None, threshold: float = 0.5) -> dict:
        pil_image = _to_pil(image)
        if box is None and bbox_fn is not None:
            box = bbox_fn(pil_image)

        inputs = processor(
            images=pil_image,
            text=text_prompt,
            input_boxes=[[box]] if box is not None else None,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        if len(result["masks"]) == 0:
            return _empty_result(pil_image)

        raw_masks = result["masks"].cpu().numpy()
        masks = np.stack([_postprocess_mask(m) if postprocess else m.astype(bool) for m in raw_masks])
        boxes = np.array([_mask_to_bbox(m) or [0, 0, *pil_image.size] for m in masks], dtype=np.float32)
        return {
            "boxes": boxes,
            "scores": result["scores"].cpu().numpy(),
            "masks": masks,
            "num_detections": len(masks),
            "image": pil_image,
        }

    return predict


def _mask_to_bbox(mask_2d: np.ndarray) -> Optional[list]:
    ys, xs = np.where(mask_2d)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


# ---------------------------------------------------------------------------
# MedSAM3 (native sam3 + LoRA)
# ---------------------------------------------------------------------------

def _default_medsam3_lora_config() -> LoRAConfig:
    return LoRAConfig(rank=16, alpha=32, dropout=0.0)


def _ensure_bpe_vocab() -> str:
    import sam3

    bpe_path = Path(sam3.__file__).parent / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not bpe_path.exists():
        import urllib.request

        url = "https://raw.githubusercontent.com/facebookresearch/sam3/main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        bpe_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[medsam3] downloading missing BPE vocab -> {bpe_path}")
        urllib.request.urlretrieve(url, bpe_path)
    return str(bpe_path)


def _resolve_lora_path(lora_weights_path: Optional[PathLike]) -> Path:
    if lora_weights_path is None:
        lora_weights_path = _DEFAULT_MEDSAM3_LORA_REPO

    path = Path(lora_weights_path)
    if path.is_file():
        return path

    from huggingface_hub import snapshot_download

    snapshot_dir = Path(snapshot_download(repo_id=str(lora_weights_path)))
    weights = next(snapshot_dir.glob("*.pt"), None)
    if weights is None:
        raise FileNotFoundError(f"No .pt file found in {snapshot_dir}")
    return weights


def _xyxy_pixels_to_cxcywh_normalized(box: list, width: int, height: int) -> list:
    x1, y1, x2, y2 = box
    return [(x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height]


def build_medsam3_predictor(
    lora_weights_path: Optional[PathLike] = None,
    lora_config: Optional[LoRAConfig] = None,
    device: Optional[torch.device] = None,
    postprocess: bool = True,
) -> PredictFn:
    """Build a MedSAM3 (native sam3 + LoRA) predictor."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = device or _pick_device()
    weights_path = _resolve_lora_path(lora_weights_path)

    model = build_sam3_image_model(
        device=str(device), load_from_HF=True, bpe_path=_ensure_bpe_vocab(), eval_mode=True,
    )
    model = apply_lora_to_model(model, lora_config or _default_medsam3_lora_config())
    load_lora_weights(model, str(weights_path))
    model.to(device).eval()
    print(f"[medsam3] ready on {device} (weights: {weights_path})")

    @torch.inference_mode()
    def predict(
        image,
        text_prompt: Optional[str] = None,
        box: Optional[list] = None,
        threshold: float = 0.5,
    ) -> dict:
        if text_prompt is None and box is None:
            raise ValueError("Provide text_prompt and/or box.")

        pil_image = _to_pil(image)
        processor = Sam3Processor(model, device=str(device), confidence_threshold=threshold)
        state = processor.set_image(pil_image)
        if text_prompt is not None:
            state = processor.set_text_prompt(text_prompt, state)
        if box is not None:
            w, h = pil_image.size
            cxcywh = _xyxy_pixels_to_cxcywh_normalized(box, w, h)
            state = processor.add_geometric_prompt(cxcywh, label=True, state=state)

        if state["masks"].shape[0] == 0:
            return _empty_result(pil_image)

        raw_masks = state["masks"].squeeze(1).cpu().numpy()  # (N, H, W) bool
        masks = np.stack([_postprocess_mask(m) if postprocess else m for m in raw_masks])
        return {
            "boxes": state["boxes"].cpu().numpy(),
            "scores": state["scores"].cpu().numpy(),
            "masks": masks,
            "num_detections": len(masks),
            "image": pil_image,
        }

    return predict