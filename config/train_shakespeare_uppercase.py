"""
Configuration for Shakespeare model training with UPPERCASE only (ablation).
Case normalized to uppercase — spaces and punctuation preserved.
Same architecture as other conditions for fair comparison.
"""

# Model architecture - same as all conditions
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Data - uppercase ablation
dataset = 'shakespeare_uppercase'

# Training - same as all conditions
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 50
eval_iters = 200
log_interval = 1

# Checkpointing
out_dir = 'out-shakespeare-uppercase'
always_save_checkpoint = True

# Learning rate schedule - same
learning_rate = 1e-3
min_lr = 1e-4
lr_decay_iters = 5000
warmup_iters = 100

# Regularization - same
weight_decay = 1e-1

# System
device = 'cuda'
compile = False

# Logging
log_file = True
wandb_log = True
wandb_project = 'scriptio-continua'
wandb_run_name = 'shakespeare-uppercase'
