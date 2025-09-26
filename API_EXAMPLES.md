# GateL0RD Training API Examples

The training API has been separated into two distinct scripts for cleaner usage:

## Single Model Training (`train.py`)

For training individual GateL0RD models with specific configurations:

```bash
# Basic training with default configuration
python train.py

# Train specific version with custom dataset
python train.py --model.version v2 --data.dataset_name FetchPickAndPlace

# Custom training parameters
python train.py --training.max_epochs 100 --model.hidden_size 128 --training.learning_rate 0.001

# Use configuration file
python train.py --config config/baseline_training.yaml

# Override config file parameters
python train.py --config config/baseline_training.yaml --training.max_epochs 200
```

## Hyperparameter Optimization (`train_hpo.py`)

For running HPO experiments using Optuna:

```bash
# Parameter scaling experiment (main use case)
python train_hpo.py --experiment_type scaling --n_trials 100

# Architecture comparison across all versions
python train_hpo.py --experiment_type comparison --n_trials 100

# Custom base version for scaling
python train_hpo.py --experiment_type scaling --base_version v0 --n_trials 50

# With timeout and custom study name
python train_hpo.py --experiment_type scaling --n_trials 200 --timeout 7200 --study_name my_scaling_study

# Use configuration file
python train_hpo.py --config config/scaling_experiment.yaml

# Custom dataset and quick parameters
python train_hpo.py --data.dataset_name BilliardBall --training.max_epochs 30 --n_trials 25
```

## Key API Changes

### Before (Old API):
- Single `train.py` script with `--experiment_type` parameter
- Mixed training and HPO logic in one file
- Complex argument parsing with many unused options for each mode

### After (New API):
- **`train.py`**: Simple, focused single model training
- **`train_hpo.py`**: Dedicated HPO experiments with Optuna
- Clean separation of concerns
- Focused argument parsing for each use case
- Removed unused `GateL0RDMultiVersionModule` class

### Benefits:
1. **Clarity**: Each script has a single, clear purpose
2. **Simplicity**: Reduced complexity in argument parsing and logic flow
3. **Performance**: Faster startup time (no HPO imports for regular training)
4. **Maintainability**: Easier to modify and extend each component independently
5. **Usability**: More intuitive command-line interface

### Configuration Compatibility:
- Existing configuration files work with both scripts
- Configuration overrides work the same way
- HPO-specific configs only needed for `train_hpo.py`