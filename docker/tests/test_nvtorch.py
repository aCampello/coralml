#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

print("Testing Torch:")

n_device = torch.cuda.device_count()
print(f" * Number of GPUS: {n_device}")

print(" * Available devices:")
for i in range(n_device):
    d_name = torch.cuda.get_device_name(i)
    print("     {i}: {d_name}")
