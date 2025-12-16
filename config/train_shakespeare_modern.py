"""
Configuration for small Shakespeare model training with MODERN text (spaces/punctuation).
Same architecture as scriptio continua version for fair comparison.
"""

# Model architecture - same as scriptio continua for fair comparison
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

# Data - MODERN text (with spaces and punctuation)
dataset = 'shakespeare_modern'

# Training - same as scriptio continua
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 50
eval_iters = 200
log_interval = 1

# Checkpointing - separate output dir
out_dir = 'out-shakespeare-modern'
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

# Logging - same project, different run name for comparison
log_file = True
wandb_log = True
wandb_project = 'scriptio-continua'  # Same project for comparison
wandb_run_name = 'shakespeare-modern'  # Different name to distinguish
