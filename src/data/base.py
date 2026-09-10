"""
data/base.py — shared data utilities for all datasets.

Every dataset loader (spleenex, deepspv, roboflow) returns images and masks
in the same format, using the functions here.

Output format (always):
    images : np.ndarray  (N, H, W, 1)  float32  values in [0, 1]
    masks  : np.ndarray  (N, H, W, 1)  float32  values in {0, 1}
"""

import json
import os
from datetime import datetime

from config import IMAGE_SIZE
import numpy as np
from PIL import Image


def resize_image(image_array, target_size=IMAGE_SIZE):
    """
    Resize a single grayscale image to the target size.
    """
    images = Image.fromarray(image_array)
    resized = images.resize(
        (target_size[1], target_size[0]),  # PIL expects (width, height)
        resample=Image.BILINEAR,
    )
    return np.array(resized)


def resize_mask(mask_array, target_size=IMAGE_SIZE):
    """
    Resize a single binary mask to the target size.
    Uses nearest-neighbor interpolation so mask values stay binary (0 or 1).
    """
    masks = Image.fromarray(mask_array)
    resized = masks.resize(
        (target_size[1], target_size[0]),  # PIL expects (width, height)
        resample=Image.NEAREST,
    )
    return np.array(resized)


def normalize_image(image_array):
    """
    Normalize a grayscale image to float32 in [0, 1].
    """
    return image_array.astype(np.float32) / 255.0


def binarize_mask(mask_array):
    """
    Convert a mask to binary float32 (0.0 or 1.0).

    Pixels with value > 127 become 1.0 (foreground).
    Pixels with value <= 127 become 0.0 (background).
    """
    return (mask_array > 127).astype(np.float32)


def add_channel_axis(array):
    """
    Add a trailing channel axis to a 2D array.

    (H, W) → (H, W, 1)
    """
    return array[..., np.newaxis]


# def shuffle_arrays(images, masks, case_ids, sources, seed=42):
#     """
#     Shuffle images, masks, and case_ids, sources together in the same random order.
#     images, masks, case_ids, sources : all shuffled in the same order
#     """
#     rng = np.random.default_rng(seed)
#     indices = rng.permutation(len(images))
#     return (
#         images[indices],
#         masks[indices],
#         [case_ids[i] for i in indices],
#         [sources[i]  for i in indices],
#     )

def shuffle_arrays(images, masks, case_ids, sources, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(images))
    return (
        images[indices],
        masks[indices],
        [case_ids[i] for i in indices],
        [sources[i]  for i in indices],
    )

def split_train_test(images, masks, case_ids, sources, test_fraction=0.15, log_name=None, seed=42):
    """
    Split images and masks into train and test sets.

    The first (1 - test_fraction) of the data becomes train,
    and the last test_fraction becomes test.

    Call shuffle_arrays first if you want a random split.
    """
    n_total = len(images)
    n_test = max(1, int(round(n_total * test_fraction)))
    n_train = n_total - n_test

    X_train, y_train = images[:n_train], masks[:n_train]
    X_test,  y_test  = images[n_train:], masks[n_train:]
 
    print(f"Split: {n_train} train / {n_test} test (total {n_total})")

    if log_name is not None:
        save_split_log(
            dataset_name=log_name,
            seed=seed,
            train_ids=case_ids[:n_train],
            train_sources=sources[:n_train],
            test_ids=case_ids[n_train:],
            test_sources=sources[n_train:],
        )
 
    return X_train, y_train, X_test, y_test


def save_split_log(dataset_name, seed, train_ids, train_sources, test_ids, test_sources):
    """
    Save a record of exactly which samples went into train vs test.
    Saved to RUNS_ROOT/splits/{dataset_name}_seed{seed}.json

    Format:
    {
      "dataset": "spleenex",
      "seed": 42,
      "train": [{"id": "1", "source": "spleenex"}, ...],
      "test":  [{"id": "7", "source": "spleenex"}, ...]
    }
    """
    from config import RUNS_ROOT
 
    splits_dir = os.path.join(RUNS_ROOT, "splits")
    os.makedirs(splits_dir, exist_ok=True)
 
    log = {
        "dataset":    dataset_name,
        "seed":       seed,
        "timestamp":  datetime.now().isoformat(),
        "n_train":    len(train_ids),
        "n_test":     len(test_ids),
        "train": [{"id": i, "source": s} for i, s in zip(train_ids,  train_sources)],
        "test":  [{"id": i, "source": s} for i, s in zip(test_ids,   test_sources)],
    }
    
    save_path = os.path.join(splits_dir, f"{dataset_name}_seed{seed}.json")
    with open(save_path, "w") as f:
        json.dump(log, f, indent=2)
 
    print(f"Split log saved: {save_path}")
 
 
def combine_datasets(*datasets):
    """
    Combine multiple datasets into one array. Does NOT shuffle or split.
 
    Example
    -------
    from data.spleenex import load_spleenex
    from data.deepspv  import load_deepspv
    from data.base     import combine_datasets, shuffle_arrays, split_train_test
 
    images, masks, case_ids = combine_datasets(
        load_spleenex(split=False),
        load_deepspv(split=False),
    )
    images, masks, case_ids = shuffle_arrays(images, masks, case_ids, seed=42)
    X_train, y_train, X_test, y_test, train_ids, test_ids = split_train_test(
        images, masks, case_ids, log_name="spleenex_deepspv", seed=42
    )
    """
    all_images, all_masks, all_ids, all_sources = [], [], [], []
 
    for images, masks, case_ids, sources in datasets:
        all_images.append(images)
        all_masks.append(masks)
        all_ids.extend(case_ids)
        all_sources.extend(sources)
 
    images = np.concatenate(all_images, axis=0)
    masks  = np.concatenate(all_masks,  axis=0)
 
    print(f"Combined: {len(images)} total samples")
    return images, masks, all_ids, all_sources
 