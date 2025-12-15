"""
Configuration for small Shakespeare model training.
Designed for quick iteration on consumer hardware (AMD Radeon 8060S or similar).

This config trains a ~10M parameter model in ~10-20 minutes.
"""

# Model architecture - small but capable
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Training
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 250
eval_iters = 200
log_interval = 10

# Learning rate schedule
learning_rate = 1e-3
min_lr = 1e-4
lr_decay_iters = 5000
warmup_iters = 100

# Regularization
weight_decay = 1e-1

# System
device = 'cuda'  # Will use ROCm on AMD GPUs
compile = False  # torch.compile can be buggy on ROCm, disable for safety

# Logging
wandb_log = False  # Set True if you have wandb configured
wandb_project = 'scriptio-continua'
