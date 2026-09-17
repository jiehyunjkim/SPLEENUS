from data.base import combine_datasets, shuffle_arrays, split_train_test
from data.deepspv import load_deepspv
from data.roboflow import load_roboflow
from data.spleenex import load_spleenex
 
# Name (as used in `train=[...]` / `test=[...]`) -> loader function.
_REGISTRY = {
    "Spleenex": load_spleenex,
    "DeepSPV": load_deepspv,
    "Roboflow": load_roboflow,
}
 
 
def _check_names(names):
    unknown = sorted(set(names) - set(_REGISTRY))
    if unknown:
        raise ValueError(
            f"Unknown dataset name(s): {unknown}. Known datasets: {sorted(_REGISTRY)}"
        )
    if not names:
        raise ValueError("Need at least one dataset name.")
 
 
def _combine(names, randomize, seed):
    """Load + concatenate the named datasets. Keeps case_ids/sources (not
    just images/masks) so a val split downstream can still write a proper
    `save_split_log`."""
    parts = [_REGISTRY[name](split=False) for name in names]
    images, masks, case_ids, sources = combine_datasets(*parts)
    if randomize:
        images, masks, case_ids, sources = shuffle_arrays(
            images, masks, case_ids, sources, seed=seed
        )
    return images, masks, case_ids, sources
 
 
class _DataNamespace:
    def __init__(self):
        self._train = None
        self._val = None
        self._test = None
        self.train_names = None
        self.test_names = None
 
    def setup(self, train, test, val_fraction=0.0, randomize=False, seed=42, log_name=None):
        """train / test: lists of dataset names, e.g. ['Spleenex', 'DeepSPV'].
        """
        _check_names(train)
        _check_names(test)
 
        self.train_names, self.test_names = list(train), list(test)
        images, masks, case_ids, sources = _combine(train, randomize, seed)
 
        if val_fraction > 0:
            X_train, y_train, X_val, y_val = split_train_test(
                images, masks, case_ids, sources,
                test_fraction=val_fraction,
                log_name=log_name or "_".join(sorted(self.train_names)),
                seed=seed,
            )
            self._train = (X_train, y_train)
            self._val = (X_val, y_val)
        else:
            self._train = (images, masks)
            self._val = None
 
        test_images, test_masks, _, _ = _combine(test, randomize, seed)
        self._test = (test_images, test_masks)
 
        val_note = f", val={len(self._val[0])}" if self._val else ""
        print(
            f"[SPLEENUS.DATA] train={self.train_names} ({len(self._train[0])} samples{val_note}) "
            f"| test={self.test_names} ({len(self._test[0])} samples)"
        )
 
 
# Single shared instance -- `S.DATA` in `SPLEENUS/__init__.py` is this object.
DATA = _DataNamespace()
 
 
def get_train():
    if DATA._train is None:
        raise RuntimeError("Call SPLEENUS.DATA.setup(...) before get_train().")
    return DATA._train
 
 
def get_val():
    if DATA._val is None:
        raise RuntimeError(
            "No validation split available -- call SPLEENUS.DATA.setup(..., val_fraction=>0) if you need one."
        )
    return DATA._val
 
 
def get_test():
    if DATA._test is None:
        raise RuntimeError("Call SPLEENUS.DATA.setup(...) before get_test().")
    return DATA._test
 