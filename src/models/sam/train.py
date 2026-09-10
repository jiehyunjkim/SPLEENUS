"""Fine-tuning for SAM3 (HF, mask-decoder only) and MedSAM3 (native + LoRA)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image as PILImage
from sklearn.model_selection import train_test_split
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from .inference import (
    _DEFAULT_MEDSAM3_LORA_REPO,
    _HF_MODEL_ID,
    _default_medsam3_lora_config,
    _ensure_bpe_vocab,
    _pick_device,
    _resolve_lora_path,
)
from .lora import LoRAConfig, apply_lora_to_model, count_parameters, load_lora_weights, save_lora_weights

PathLike = Union[str, Path]


# ===========================================================================
# SAM3 (HuggingFace) -- mask decoder only
# ===========================================================================

_SAM3_CHECKPOINT_NAME = "sam3_finetuned_weights.safetensors"


def _tight_box_from_mask(mask: np.ndarray) -> np.ndarray:
    """cv2 bounding rect of the mask's largest contour."""
    binary = np.ascontiguousarray((mask > 0).astype(np.uint8))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        return np.array([x, y, x + w, y + h])
    return np.array([0, 0, mask.shape[1], mask.shape[0]])


class _SAM3Dataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, boxes: Optional[np.ndarray]):
        self.images, self.masks, self.boxes = images, masks, boxes

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, i: int):
        mask = self.masks[i]
        mask = mask[..., 0] if mask.ndim == 3 else mask
        box = self.boxes[i] if self.boxes is not None else None
        return self.images[i], mask, box


def _sam3_metrics(device):
    from torchmetrics import MeanMetric, MetricCollection
    from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex

    return MetricCollection({"dice": BinaryF1Score(), "iou": BinaryJaccardIndex(), "loss": MeanMetric()}).to(device)


class _SAM3Trainer:
    def __init__(self, images, masks, boxes, *, output_dir, epochs, lr, val_split, seed, text_prompt, device):
        from transformers import Sam3Model, Sam3Processor

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.device = device or _pick_device()
        self.text_prompt = text_prompt
        self.use_boxes = boxes is not None

        self.processor = Sam3Processor.from_pretrained(_HF_MODEL_ID)
        self.model = Sam3Model.from_pretrained(_HF_MODEL_ID).to(self.device)
        for name, p in self.model.named_parameters():
            p.requires_grad = "mask_decoder" in name
        self._trainable_keys = {n for n, p in self.model.named_parameters() if p.requires_grad}
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[sam3-train] device={self.device} trainable_params={n_trainable:,} "
              f"prompt={'text+box' if self.use_boxes else 'text-only'}")

        self.optimizer = Adam([p for p in self.model.parameters() if p.requires_grad], lr=lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

        dataset = _SAM3Dataset(images, masks, boxes)
        self.train_loader, self.val_loader = self._split(dataset, val_split, seed)
        self.train_metrics, self.val_metrics = _sam3_metrics(self.device), _sam3_metrics(self.device)
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(self.output_dir / "tb"))
        except Exception:
            self.writer = None
        self.global_step = 0
        self.best_val_dice = -1.0

    @staticmethod
    def _split(dataset, val_split, seed):
        n = len(dataset)
        n_val = max(1, round(n * val_split)) if val_split > 0 else 0
        if n_val == 0 or n_val >= n:
            return DataLoader(dataset, batch_size=1, shuffle=True), None
        idx = np.arange(n)
        train_idx, val_idx = train_test_split(idx, test_size=n_val, random_state=seed)
        return (
            DataLoader(Subset(dataset, train_idx.tolist()), batch_size=1, shuffle=True),
            DataLoader(Subset(dataset, val_idx.tolist()), batch_size=1, shuffle=False),
        )

    def _forward(self, batch):
        images, masks, boxes = batch
        targets = masks.float().unsqueeze(1).to(self.device)
        proc_kwargs = {"text": [self.text_prompt] * len(images)}
        if self.use_boxes:
            proc_kwargs["input_boxes"] = boxes.unsqueeze(1).tolist()
        inputs = self.processor(images=[img.numpy() for img in images], return_tensors="pt", **proc_kwargs).to(self.device)
        outputs = self.model(**inputs)
        logits = F.interpolate(outputs.pred_masks[:, :1], size=targets.shape[-2:], mode="bilinear", align_corners=False)
        return logits, targets

    def _run_epoch(self, loader, metrics, train: bool):
        metrics.reset()
        last_loss = None
        for batch in loader:
            logits, targets = self._forward(batch)
            loss = self.loss_fn(logits, targets)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.writer is not None:
                    self.writer.add_scalar("train/step_loss", loss.item(), self.global_step)
                self.global_step += 1
            probs = torch.sigmoid(logits).detach()
            metrics["dice"].update(probs, targets.long())
            metrics["iou"].update(probs, targets.long())
            metrics["loss"].update(loss.detach())
            last_loss = loss
        return {k: v.item() for k, v in metrics.compute().items()}

    def save(self, path: Path) -> Path:
        from safetensors.torch import save_file

        full = self.model.state_dict()
        trainable = {k: v.detach().cpu().contiguous() for k, v in full.items() if k in self._trainable_keys}
        save_file(trainable, str(path))
        return path

    def fit(self) -> Path:
        best_path = self.output_dir / _SAM3_CHECKPOINT_NAME
        last_path = self.output_dir / ("last_" + _SAM3_CHECKPOINT_NAME)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_stats = self._run_epoch(self.train_loader, self.train_metrics, train=True)
            val_stats = {}
            if self.val_loader is not None:
                self.model.eval()
                with torch.inference_mode():
                    val_stats = self._run_epoch(self.val_loader, self.val_metrics, train=False)
            if self.writer is not None:
                for k, v in train_stats.items():
                    self.writer.add_scalar(f"train/{k}", v, epoch)
                for k, v in val_stats.items():
                    self.writer.add_scalar(f"val/{k}", v, epoch)
            msg = f"[sam3-train][epoch {epoch}/{self.epochs}] " + ", ".join(f"train_{k}={v:.4f}" for k, v in train_stats.items())
            if val_stats:
                msg += " | " + ", ".join(f"val_{k}={v:.4f}" for k, v in val_stats.items())
            print(msg)

            self.save(last_path)
            if val_stats and val_stats["dice"] > self.best_val_dice:
                self.best_val_dice = val_stats["dice"]
                self.save(best_path)
        if self.val_loader is None:
            import shutil

            shutil.copyfile(last_path, best_path)
        if self.writer is not None:
            self.writer.close()
        return best_path


def train_sam3(
    images: np.ndarray,
    masks: np.ndarray,
    *,
    output_dir: PathLike,
    epochs: int = 20,
    lr: float = 1e-4,
    val_split: float = 0.1,
    seed: int = 42,
    box_source: str = "gt",
    boxes: Optional[np.ndarray] = None,
    text_prompt: str = "spleen",
    device: Optional[torch.device] = None,
) -> Path:
    """Fine-tune SAM3's mask decoder on in-memory (images, masks)."""
    if boxes is None:
        if box_source == "gt":
            boxes = np.stack([_tight_box_from_mask(m) for m in masks])
        elif box_source == "none":
            boxes = None
        else:
            raise ValueError(f"box_source must be 'gt' or 'none', got {box_source!r}")

    trainer = _SAM3Trainer(
        images, masks.astype(np.uint8), boxes,
        output_dir=output_dir, epochs=epochs, lr=lr, val_split=val_split, seed=seed,
        text_prompt=text_prompt, device=device,
    )
    return trainer.fit()


# ===========================================================================
# MedSAM3 (native sam3 + LoRA)
# ===========================================================================

class _MedSAM3Dataset(Dataset):
    """Builds `Datapoint`s straight from in-memory (image, mask) arrays."""

    def __init__(self, images, masks, category, resolution=1008, use_box_prompt=False):
        self.images, self.masks, self.category, self.resolution = images, masks, category, resolution
        self.use_box_prompt = use_box_prompt
        from torchvision.transforms import v2

        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata, Object

        pil_image = PILImage.fromarray(self.images[idx]).convert("RGB")
        orig_w, orig_h = pil_image.size
        pil_image_resized = pil_image.resize((self.resolution, self.resolution), PILImage.BILINEAR)
        image_tensor = self.transform(pil_image_resized)

        mask = (self.masks[idx] > 0).astype(np.uint8)
        ys, xs = np.where(mask)
        objects = []
        if len(xs) > 0:
            x, y, w, h = int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min())
            sx, sy = self.resolution / orig_w, self.resolution / orig_h
            # normalized CxCyWH, matching what the rest of the native pipeline expects
            box = torch.tensor([
                (x + w / 2) * sx / self.resolution,
                (y + h / 2) * sy / self.resolution,
                w * sx / self.resolution,
                h * sy / self.resolution,
            ], dtype=torch.float32)
            seg = torch.nn.functional.interpolate(
                torch.from_numpy(mask)[None, None].float(), size=(self.resolution, self.resolution), mode="nearest",
            )[0, 0] > 0.5
            objects.append(Object(bbox=box, area=(box[2] * box[3]).item(), object_id=0, segment=seg))

        input_bbox, input_bbox_label = None, None
        if self.use_box_prompt and objects:
            input_bbox = box.unsqueeze(0)  # (1, 4) -- same normalized CxCyWH box as the Object above
            input_bbox_label = torch.ones(1, dtype=torch.bool)  # 1 positive box

        query = FindQueryLoaded(
            query_text=self.category,
            image_id=0,
            object_ids_output=[o.object_id for o in objects],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=idx, original_image_id=idx, original_category_id=1,
                original_size=(orig_h, orig_w), object_id=-1, frame_index=-1,
            ),
        )
        return Datapoint(
            find_queries=[query],
            images=[SAMImage(data=image_tensor, objects=objects, size=(self.resolution, self.resolution))],
            raw_images=[pil_image_resized],
        )


class _MedSAM3Trainer:
    def __init__(
        self, images, masks, category, *, output_dir, epochs, batch_size, lr, weight_decay,
        val_split, seed, lora_config, pretrained_lora, device, box_source="none",
    ):
        from sam3.model.model_misc import SAM3Output
        from sam3.model_builder import build_sam3_image_model
        from sam3.train.data.collator import collate_fn_api
        from sam3.train.loss.loss_fns import CORE_LOSS_KEY, Boxes, IABCEMdetr, Masks
        from sam3.train.loss.sam3_loss import Sam3LossWrapper
        from sam3.train.matcher import BinaryHungarianMatcherV2, BinaryOneToManyMatcher

        self.device = device or _pick_device()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self._SAM3Output = SAM3Output
        self._CORE_LOSS_KEY = CORE_LOSS_KEY
        self._collate = lambda batch: collate_fn_api(batch, dict_key="input", with_seg_masks=True)

        print(f"[medsam3-train] building model on {self.device}...")
        self.model = build_sam3_image_model(
            device=str(self.device), compile=False, load_from_HF=True,
            bpe_path=_ensure_bpe_vocab(), eval_mode=False,
        )
        self.model = apply_lora_to_model(self.model, lora_config or _default_medsam3_lora_config())
        if pretrained_lora:
            repo_or_path = None if pretrained_lora is True else pretrained_lora
            weights_path = _resolve_lora_path(repo_or_path)
            load_lora_weights(self.model, str(weights_path))
            print(f"[medsam3-train] warm-started from {weights_path}")
        stats = count_parameters(self.model)
        print(f"[medsam3-train] trainable: {stats['trainable_parameters']:,} ({stats['trainable_percentage']:.2f}%)")
        self.model.to(self.device)

        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
        self.matcher = BinaryHungarianMatcherV2(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, focal=True)
        o2m_matcher = BinaryOneToManyMatcher(alpha=0.3, threshold=0.4, topk=4)
        self.loss_wrapper = Sam3LossWrapper(
            loss_fns_find=[
                Boxes(weight_dict={"loss_bbox": 5.0, "loss_giou": 2.0}),
                IABCEMdetr(
                    pos_weight=10.0, weight_dict={"loss_ce": 20.0, "presence_loss": 20.0},
                    pos_focal=False, alpha=0.25, gamma=2, use_presence=True, pad_n_queries=200,
                ),
                Masks(weight_dict={"loss_mask": 200.0, "loss_dice": 10.0}, focal_alpha=0.25, focal_gamma=2.0, compute_aux=False),
            ],
            matcher=self.matcher, o2m_matcher=o2m_matcher, o2m_weight=2.0,
            use_o2m_matcher_on_o2m_aux=False, normalization="local", normalize_by_valid_object_num=False,
        )

        n = len(images)
        n_val = max(1, round(n * val_split)) if val_split > 0 else 0
        idx = np.arange(n)
        if 0 < n_val < n:
            train_idx, val_idx = train_test_split(idx, test_size=n_val, random_state=seed)
        else:
            train_idx, val_idx = idx, np.array([], dtype=int)


        if box_source not in ("none", "gt"):
            raise ValueError(f"box_source must be 'none' or 'gt', got {box_source!r}")
        full_ds = _MedSAM3Dataset(images, masks, category, use_box_prompt=(box_source == "gt"))
        self.train_loader = DataLoader(
            Subset(full_ds, train_idx.tolist()), batch_size=batch_size, shuffle=True, collate_fn=self._collate,
        )
        self.val_loader = (
            DataLoader(Subset(full_ds, val_idx.tolist()), batch_size=batch_size, shuffle=False, collate_fn=self._collate)
            if len(val_idx) else None
        )
        print(f"[medsam3-train] train={len(train_idx)} val={len(val_idx)}")

    @staticmethod
    def _to_device(obj, device):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(_MedSAM3Trainer._to_device(x, device) for x in obj)
        if isinstance(obj, dict):
            return {k: _MedSAM3Trainer._to_device(v, device) for k, v in obj.items()}
        if hasattr(obj, "__dataclass_fields__"):
            for f in obj.__dataclass_fields__:
                setattr(obj, f, _MedSAM3Trainer._to_device(getattr(obj, f), device))
            return obj
        return obj

    def _step(self, batch_dict, train: bool) -> float:
        input_batch = self._to_device(batch_dict["input"], self.device)
        outputs_list = self.model(input_batch)
        find_targets = [self.model.back_convert(t) for t in input_batch.find_targets]
        for targets in find_targets:
            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(self.device)

        with self._SAM3Output.iteration_mode(outputs_list, iter_mode=self._SAM3Output.IterMode.ALL_STEPS_PER_STAGE) as it:
            for stage_outputs, stage_targets in zip(it, find_targets):
                for outputs in stage_outputs:
                    outputs["indices"] = self.matcher(outputs, stage_targets)
                    for aux in outputs.get("aux_outputs", []):
                        aux["indices"] = self.matcher(aux, stage_targets)

        loss_dict = self.loss_wrapper(outputs_list, find_targets)
        loss = loss_dict[self._CORE_LOSS_KEY]
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def fit(self) -> Path:
        best_path = self.output_dir / "best_lora_weights.pt"
        last_path = self.output_dir / "last_lora_weights.pt"
        best_val = float("inf")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses = [self._step(b, train=True) for b in tqdm(self.train_loader, desc=f"[medsam3-train] epoch {epoch}")]
            avg_train = sum(train_losses) / len(train_losses)

            avg_val = None
            if self.val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    val_losses = [self._step(b, train=False) for b in self.val_loader]
                avg_val = sum(val_losses) / len(val_losses)

            msg = f"[medsam3-train][epoch {epoch}/{self.epochs}] train_loss={avg_train:.4f}"
            print(msg + (f" val_loss={avg_val:.4f}" if avg_val is not None else ""))

            save_lora_weights(self.model, str(last_path))
            if avg_val is None or avg_val < best_val:
                best_val = avg_val if avg_val is not None else best_val
                save_lora_weights(self.model, str(best_path))

        return best_path


def train_medsam3(
    images: np.ndarray,
    masks: np.ndarray,
    output_dir: PathLike,
    *,
    category: str = "spleen",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    val_split: float = 0.2,
    seed: int = 42,
    lora_config: Optional[LoRAConfig] = None,
    pretrained_lora: Union[str, bool, None] = _DEFAULT_MEDSAM3_LORA_REPO,
    device: Optional[torch.device] = None,
    box_source: str = "none",
) -> Path:
    """Fine-tune MedSAM3's LoRA adapters on in-memory (images, masks)."""
    trainer = _MedSAM3Trainer(
        images, masks, category,
        output_dir=output_dir, epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
        val_split=val_split, seed=seed, lora_config=lora_config, pretrained_lora=pretrained_lora, device=device,
        box_source=box_source,
    )
    return trainer.fit()