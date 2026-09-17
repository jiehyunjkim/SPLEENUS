# Name (as used in `MODELS.get(...)`) -> (constructor, extra required kwargs).
# "unet"/"unet++"/"unet3++" all go through UNetSegmentor(model_name=...).
_UNET_NAMES = {"unet", "unet++", "unet3++"}
 
 
class _ModelsNamespace: 
    def get(self, name, **kwargs):
        """name: one of "unet", "unet++", "unet3++", "nnunet", "sam3", "medsam3".
        """
        if name in _UNET_NAMES:
            from models.unet import UNetSegmentor
            return UNetSegmentor(name, **kwargs)
        if name == "nnunet":
            if "dataset_name" not in kwargs or "dataset_id" not in kwargs:
                raise ValueError(
                    "MODELS.get('nnunet', ...) needs dataset_name= and dataset_id=, "
                    "e.g. MODELS.get('nnunet', dataset_name='Dataset103_SpleenUS_DeepSPV', dataset_id=103)"
                )
            from models.nnunet import NNUNetSegmentor
            return NNUNetSegmentor(**kwargs)
        if name == "sam3":
            from models.sam_models import SAM3Segmentor
            return SAM3Segmentor(**kwargs)
        if name == "medsam3":
            from models.sam_models import MedSAM3Segmentor
            return MedSAM3Segmentor(**kwargs)
        raise ValueError(
            f"Unknown model '{name}'. Choose from: "
            f"{sorted(_UNET_NAMES | {'nnunet', 'sam3', 'medsam3'})}"
        )
 
 
MODELS = _ModelsNamespace()
