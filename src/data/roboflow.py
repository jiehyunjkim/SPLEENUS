"""
data/roboflow.py — loader for the Roboflow spleen ultrasound dataset.

Files expected:
    ROBOFLOW/
        images.npz   — array key "images", shape (N, 300, 300, 3), uint8
        masks.npz    — array key "masks",  shape (N, 300, 300),    uint8, values {0, 1}

Preprocessing matches the original Team 1 pipeline:
    crop top 24 rows → take channel 0 → normalize → pad to 320×320
"""
import os
import numpy as np

from data.base import (
    add_channel_axis,
    shuffle_arrays,
    split_train_test,
)
from config import ROBOFLOW_ROOT, IMAGE_SIZE


# Roboflow-specific preprocessing constants (from Team 1 pipeline)
SOURCE         = "roboflow"
ROWS_TO_CROP   = 24   # equipment overlay at the top of each frame
ORIGINAL_H     = 300  # height after crop: 300 - 24 = 276
ORIGINAL_W     = 300
CROPPED_H      = ORIGINAL_H - ROWS_TO_CROP  # 276

# Padding to go from (276, 300) → IMAGE_SIZE
PAD_TOP    = (IMAGE_SIZE[0] - CROPPED_H) // 2       # 22
PAD_BOTTOM = IMAGE_SIZE[0] - CROPPED_H - PAD_TOP    # 22
PAD_LEFT   = (IMAGE_SIZE[1] - ORIGINAL_W) // 2      # 10
PAD_RIGHT  = IMAGE_SIZE[1] - ORIGINAL_W - PAD_LEFT  # 10


def _preprocess_images(raw_images):
    """
    Apply Team 1 preprocessing to a batch of Roboflow images.

    Steps:
        1. Crop top ROWS_TO_CROP rows (remove equipment overlay)
        2. Take channel 0 only (all channels are identical for grayscale)
        3. Normalize to [0, 1]
        4. Pad to IMAGE_SIZE

    Parameters
    ----------
    raw_images : np.ndarray  (N, 300, 300, 3)  uint8

    Returns
    -------
    np.ndarray  (N, 320, 320, 1)  float32
    """
    cropped    = raw_images[:, ROWS_TO_CROP:, :, 0]           # (N, 276, 300)
    normalized = cropped.astype(np.float32) / 255.0           # (N, 276, 300)
    padded     = np.pad(
        normalized,
        ((0, 0), (PAD_TOP, PAD_BOTTOM), (PAD_LEFT, PAD_RIGHT)),
        mode="constant",
        constant_values=0.0,
    )                                                          # (N, 320, 320)
    return padded[..., np.newaxis]                             # (N, 320, 320, 1)


def _preprocess_masks(raw_masks):
    """
    1. Crop top ROWS_TO_CROP rows
    2. Binarize (values are already {0, 1}, so just cast to float32)
    3. Pad to IMAGE_SIZE
    """
    cropped  = raw_masks[:, ROWS_TO_CROP:, :]                 # (N, 276, 300)
    binary   = (cropped > 0).astype(np.float32)               # (N, 276, 300)
    padded   = np.pad(
        binary,
        ((0, 0), (PAD_TOP, PAD_BOTTOM), (PAD_LEFT, PAD_RIGHT)),
        mode="constant",
        constant_values=0.0,
    )                                                          # (N, 320, 320)
    return padded[..., np.newaxis]                             # (N, 320, 320, 1)


def load_roboflow(split=True, test_fraction=0.15, seed=42):
    """
    Load the Roboflow spleen ultrasound dataset.

    Parameters
    ----------
    test_fraction : float   fraction held out for test set (default 0.15)
    seed          : int     random seed for shuffling

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        images : (N, 320, 320, 1)  float32  in [0, 1]
        masks  : (N, 320, 320, 1)  float32  in {0, 1}
    """
    images_path = os.path.join(ROBOFLOW_ROOT, "images.npz")
    masks_path  = os.path.join(ROBOFLOW_ROOT, "masks.npz")

    raw_images = np.load(images_path)["images"]  # (N, 300, 300, 3) uint8
    raw_masks  = np.load(masks_path)["masks"]    # (N, 300, 300)    uint8

    images = _preprocess_images(raw_images)
    masks  = _preprocess_masks(raw_masks)
    case_ids = [str(i) for i in range(len(images))]
    sources = [SOURCE] * len(images)

    print(f"Roboflow: loaded {len(images)} samples")
    print(f"Roboflow: spleen coverage {100 * masks.mean():.2f}%")

    images, masks, case_ids, sources = shuffle_arrays(images, masks, case_ids, sources, seed=seed)
 
    if not split:
        return images, masks, case_ids, sources
 
    return split_train_test(
        images, masks, case_ids, sources,
        test_fraction=test_fraction,
        log_name=SOURCE,
        seed=seed,
    )
