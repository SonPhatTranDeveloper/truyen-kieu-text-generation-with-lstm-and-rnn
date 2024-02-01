"""
This script confirms if PyTorch MPS version has been installed successfully
Successful message is tensor([1.], device='mps:0')
"""

import torch

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS device not found.")