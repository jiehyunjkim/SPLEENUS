import os
import random
 
import numpy as np
 
 
def setup_notebook(seed=42, env_path="/home/jiehyun.kim001/SPLEENUS/team-SAMv3-impact/.env.local", require_hf_token=True):
    """Seeds, TF determinism flags, and .env.local -- call this first,
    before importing anything from `models/` or `data/` (in particular,
    before `S.MODELS.get(...)` or `S.DATA.setup(...)`).
 
    Parameters
    ----------
    seed             : int   used for tf/np/random seeding (default 42)
    env_path         : str   path to .env.local, relative to the notebook
                              (default "../.env.local", matching examples/)
    require_hf_token : bool  if True (default), raises if HF_TOKEN isn't
                              set after loading env_path -- SAM3/MedSAM3
                              need it. Set False for notebooks that don't
                              touch SAM3/MedSAM3.
    """
    # Env vars first, before TF is ever imported/touched -- order matters here.
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # must be set before TF touches the GPU
 
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
 
    from dotenv import load_dotenv
    load_dotenv(env_path)
 
    if require_hf_token and not os.environ.get("HF_TOKEN"):
        raise AssertionError(
            "Set HF_TOKEN in your environment or .env.local before running."
        )
 
    print(f"[SPLEENUS] notebook set up (seed={seed}, env={env_path})")
 