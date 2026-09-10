"""
data/deepspv.py — loader for the DeepSPV synthetic dataset.

Directory structure expected:
    DeepSPV/256_size/
        syn_imgs/
            syn_img0001.png
            syn_img0002.png
            ...
        syn_layouts/
            syn_layout0001.png
            syn_layout0002.png
            ...

Layout values: 0 = background, 1 = other organ, 2 = spleen
"""

import glob
import os
import numpy as np
from PIL import Image

from data.base import (
    resize_image,
    resize_mask,
    normalize_image,
    add_channel_axis,
    shuffle_arrays,
    split_train_test,
)
from config import DEEPSPV_ROOT

SOURCE       = "deepspv"
SPLEEN_LABEL = 2  # value in syn_layouts that represents the spleen


def _extract_file_id(filepath):
    """Extract the numeric ID from a filename like 'syn_img0001.png' → '0001'."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    return "".join(ch for ch in basename if ch.isdigit())


def _layout_to_spleen_mask(layout_array):
    """
    Convert a DeepSPV layout array to a binary spleen mask.

    Normal case : layout holds {0, 1, 2} and spleen == 2.
    Fallback     : if 2 is not present, the highest label is treated as spleen.
    Empty case   : if only one value exists, return an all-zero mask.

    Returns uint8 array with 0 = background, 255 = spleen.
    """
    unique_values = np.unique(layout_array)

    if SPLEEN_LABEL in unique_values:
        spleen_mask = (layout_array == SPLEEN_LABEL)
    elif len(unique_values) >= 2:
        spleen_mask = (layout_array == unique_values.max())
    else:
        spleen_mask = np.zeros_like(layout_array, dtype=bool)

    return (spleen_mask * 255).astype(np.uint8)


def load_deepspv(split=True, test_fraction=0.15, seed=42):
    """
    Load all DeepSPV synthetic images and spleen masks.
 
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
        train_ids, test_ids              : list of str  (file numeric IDs)
 
    If split=False:
        images   : np.ndarray  (N, 320, 320, 1) float32
        masks    : np.ndarray  (N, 320, 320, 1) float32
        case_ids : list of str
    """
    images_dir  = os.path.join(DEEPSPV_ROOT, "syn_imgs")
    layouts_dir = os.path.join(DEEPSPV_ROOT, "syn_layouts")
 
    image_paths  = sorted(glob.glob(os.path.join(images_dir,  "*.png")))
    layout_paths = sorted(glob.glob(os.path.join(layouts_dir, "*.png")))
    layout_by_id = {_extract_file_id(p): p for p in layout_paths}
 
    images, masks, valid_ids, sources = [], [], [], []
    skipped = 0
 
    for image_path in image_paths:
        file_id     = _extract_file_id(image_path)
        layout_path = layout_by_id.get(file_id)
 
        if layout_path is None:
            skipped += 1
            continue
 
        image  = np.array(Image.open(image_path).convert("L"))
        layout = np.array(Image.open(layout_path).convert("L"))
        mask   = _layout_to_spleen_mask(layout)
 
        image = resize_image(image)
        mask  = resize_mask(mask)
        image = normalize_image(image)
        mask  = (mask > 127).astype(np.float32)
 
        images.append(add_channel_axis(image))
        masks.append(add_channel_axis(mask))
        valid_ids.append(file_id)
        sources.append(SOURCE)
 
    images = np.array(images, dtype=np.float32)
    masks  = np.array(masks,  dtype=np.float32)
 
    print(f"DeepSPV: loaded {len(images)} samples ({skipped} skipped)")
    print(f"DeepSPV: spleen coverage {100 * masks.mean():.2f}%")
 
    images, masks, valid_ids, sources = shuffle_arrays(images, masks, valid_ids, sources, seed=seed)
 
    if not split:
        return images, masks, valid_ids, sources
 
    return split_train_test(
        images, masks, valid_ids, sources,
        test_fraction=test_fraction,
        log_name=SOURCE,
        seed=seed,
    )