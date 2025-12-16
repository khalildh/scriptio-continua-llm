# Training Guide

## Start Training (Background)

Run training that continues even after closing VS Code:

```bash
# Run both models (modern then scriptio)
nohup python train_comparison.py > training_comparison.log 2>&1 &

# Or run a single model
nohup python train.py config/train_shakespeare_modern.py > modern.log 2>&1 &
nohup python train.py config/train_shakespeare_small.py > scriptio.log 2>&1 &
```

## Monitor Training

```bash
# Live log output (Ctrl+C to stop watching)
tail -f training_comparison.log

# Last 50 lines
tail -50 training_comparison.log

# Check if still running
ps aux | grep train

# Check GPU usage
rocm-smi
```

## Resume Training

If training stops (crash, interrupt, etc.), resume from checkpoint:

```bash
nohup python train.py config/train_shakespeare_modern.py --init_from=resume > modern.log 2>&1 &
```

This continues the same wandb run and picks up from the last checkpoint.

## Stop Training

```bash
# Find the process ID
ps aux | grep train

# Kill it (replace PID with actual number)
kill PID

# Or kill all training
pkill -f train.py
```

## Expected Timeline

| Model | Iterations | Time |
|-------|------------|------|
| Modern | 5000 | ~6 hours |
| Scriptio | 5000 | ~6 hours |
| **Total** | 10000 | **~12 hours** |

## Check Results

- **wandb dashboard**: Live charts at https://wandb.ai
- **Checkpoints**: `out-shakespeare-modern/` and `out-shakespeare-scriptio/`
- **Logs**: `training_comparison.log` or individual `.log` files
