"""
Configuration for medium Shakespeare model training.
~25M parameters, takes ~30-60 minutes on consumer hardware.

Use this after validating results with the small config.
"""

# Model architecture - medium
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2

# Training
batch_size = 48
block_size = 256
max_iters = 10000
eval_interval = 500
eval_iters = 200
log_interval = 10

# Learning rate schedule
learning_rate = 6e-4
min_lr = 6e-5
lr_decay_iters = 10000
warmup_iters = 200

# Regularization
weight_decay = 1e-1

# System
device = 'cuda'
compile = False

# Logging
wandb_log = False
wandb_project = 'scriptio-continua'
