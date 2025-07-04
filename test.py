# ---------------------------------------------------------------
# AIR-Mamba Model Inference Script
#
# Paper:
#   "Domain-Adaptive UAV Recognition Using IR-UWB Radar and a Lightweight Mamba-Based Network"
#   Shengyuan Li, Xinyue Dong, Yiheng Fan, Xiangwei Zhu, Xuelin Yuan
#   IEEE Sensors Journal, under review, 2024
#
# Note:
#   This code is released for academic verification purposes only.
# ---------------------------------------------------------------

import torch
from models.core import airmamba_s

if __name__ == '__main__':
    model = airmamba_s().cuda().eval()
    dummy_input = torch.randn(1, 3, 256, 256).cuda()
    out = model(dummy_input)
    print(f"Model output shape: {out.shape}")

