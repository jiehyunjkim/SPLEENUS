# src/inia/config.py

# Paths
DATA_ROOT  = "/raid/mpsych/SPLEENUS"
MODEL_ROOT = "/raid/mpsych/SPLEENUS/models"
RUNS_ROOT  = "/raid/mpsych/SPLEENUS/runs"

# Datapath
SPLEENEX_ROOT = "/raid/mpsych/SPLEENUS/spleenex"
DEEPSPV_ROOT  = "/raid/mpsych/SPLEENUS/DeepSPV/256_size"
ROBOFLOW_ROOT = "/raid/mpsych/SPLEENUS/ROBOFLOW"


# Image size for UNet family (and all models for now)
# Note: determined by training data constraints, not optimal for all datasets
# Will be revisited when real INIAcore data is available
IMAGE_SIZE = (320, 320)  # (height, width)