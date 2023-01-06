"""
    check_gpu.py

# Description
This script checks and prints the GPUs available.
Previously using torch, now checks using tensorflow.

# Authors
- Sasha Petrenko <sap625@mst.edu> @AP6YC
- Date: 6/3/2022
"""

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

# Custom imports
import tensorflow as tf

# -----------------------------------------------------------------------------
# SCRIPT
# -----------------------------------------------------------------------------

# Show the gpu devices available
print(tf.config.list_physical_devices('GPU'))

# -----------------------------------------------------------------------------
# Deprecated Torch code
# -----------------------------------------------------------------------------

# import torch

# cuda_avail = torch.cuda.is_available()
# print('CUDA Available: ', cuda_avail)
# if cuda_avail:
#     print('Number of GPUs: ', torch.cuda.device_count())
#     print('Current GPU: ', torch.cuda.current_device())