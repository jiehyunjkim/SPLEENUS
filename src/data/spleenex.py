"""
data/spleenex.py — loader for the Spleenex dataset.

Directory structure expected:
    spleenex/
        images/
            1/   1.png
            2/   2.png
            ...
        masks/
            1/   1.png
            2/   2.png
            ...

Mask values: 0 = background, 255 = spleen (binary PNG)

Usage
-----
# Split into train/test (default)
X_train, y_train, X_test, y_test, train_ids, test_ids = load_spleenex()
 
# All samples, no split (for combining with other datasets)
images, masks, case_ids = load_spleenex(split=False)
"""

import os
import numpy as np
from PIL import Image
 
from data.base import (
    resize_image, resize_mask,
    normalize_image, binarize_mask,
    add_channel_axis, shuffle_arrays,
    split_train_test,
)
from config import SPLEENEX_ROOT

SOURCE = "spleenex"
 
def load_spleenex(split=True, test_fraction=0.15, seed=42):
    """
    Load all Spleenex images and masks.
 
    Parameters
    ----------
    split         : bool   if True, returns train/test split (default)
                           if False, returns all samples unsplit
    test_fraction : float  fraction for test set (only used when split=True)
    seed          : int    random seed for shuffling
 
    Returns
    -------
    If split=True:
        X_train, y_train, X_test, y_test : np.ndarray  (N, 320, 320, 1) float32
        train_ids, test_ids              : list of str  (folder names)
 
    If split=False:
        images   : np.ndarray  (N, 320, 320, 1) float32
        masks    : np.ndarray  (N, 320, 320, 1) float32
        case_ids : list of str
    """
    images_dir = os.path.join(SPLEENEX_ROOT, "images")
    masks_dir  = os.path.join(SPLEENEX_ROOT, "masks")
 
    case_ids = sorted(os.listdir(images_dir), key=lambda x: int(x))
 
    images, masks, valid_ids, sources = [], [], [], []
    skipped = 0
 
    for case_id in case_ids:
        image_path = os.path.join(images_dir, case_id, f"{case_id}.png")
        mask_path  = os.path.join(masks_dir,  case_id, f"{case_id}.png")
 
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            skipped += 1
            continue
 
        image = np.array(Image.open(image_path).convert("L"))
        mask  = np.array(Image.open(mask_path).convert("L"))
 
        image = resize_image(image)
        mask  = resize_mask(mask)
        image = normalize_image(image)
        mask  = binarize_mask(mask)
 
        images.append(add_channel_axis(image))
        masks.append(add_channel_axis(mask))
        valid_ids.append(case_id)
        sources.append(SOURCE)
 
    images = np.array(images, dtype=np.float32)
    masks  = np.array(masks,  dtype=np.float32)
 
    print(f"Spleenex: loaded {len(images)} samples ({skipped} skipped)")
    print(f"Spleenex: spleen coverage {100 * masks.mean():.2f}%")
 
    images, masks, valid_ids, sources = shuffle_arrays(images, masks, valid_ids, sources, seed=seed)
 
    if not split:
        return images, masks, valid_ids, sources
 
    return split_train_test(
        images, masks, valid_ids, sources,
        test_fraction=test_fraction,
        log_name=SOURCE,
        seed=seed,
    )