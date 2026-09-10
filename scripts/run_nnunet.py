import os
import sys
sys.path.insert(0, '/home/jiehyun.kim001/inia/src')

from models.nnunet import NNUNetSegmentor

model = NNUNetSegmentor("Dataset103_SpleenUS_DeepSPV", 103)
model.setup("/raid/mpsych/SPLEENUS/nnunet_spl_deepspv_new")
model.fit()