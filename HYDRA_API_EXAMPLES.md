# GateL0RD Hydra Configuration API Examples

The training API now uses Hydra for configuration management, providing powerful composition, override capabilities, and structured configuration.

## Project Structure

```
conf/
├── config.yaml                 # Main configuration file
├── experiment/                 # Experiment-specific configs
│   ├── baseline_training.yaml
│   └── scaling_experiment.yaml
├── model/                      # Model configurations
│   ├── v0.yaml
│   ├── v1.yaml
│   ├── v2.yaml
│   └── v3.yaml
├── data/                       # Data configurations
│   ├── billiard_ball.yaml
│   └── fetch_pick_place.yaml
├── training/                   # Training configurations
│   ├── default.yaml
│   └── hpo.yaml
└── hpo/                        # HPO configurations
    └── scaling.yaml
```

## Single Model Training (`train_hydra.py`)

### Basic Usage
```bash
# Use default configuration
python train_hydra.py

# Use specific experiment configuration
python train_hydra.py experiment=baseline_training

# Use different model versions
python train_hydra.py model=v2
python train_hydra.py model=v3

# Use different datasets
python train_hydra.py data=fetch_pick_place
python train_hydra.py data=billiard_ball
```

### Configuration Overrides
```bash
# Override single parameters
python train_hydra.py training.max_epochs=200
python train_hydra.py model.hidden_size=128
python train_hydra.py data.batch_size=64

# Override multiple parameters
python train_hydra.py model=v2 training.max_epochs=150 model.hidden_size=256

# Override nested parameters
python train_hydra.py training.early_stopping_patience=20 training.learning_rate=0.01

# Set experiment name and description
python train_hydra.py name=my_experiment description="Testing v2 with large hidden size"
```

### Advanced Usage
```bash
# Use custom config directory
python train_hydra.py --config-path=/path/to/configs --config-name=my_config

# Save outputs to specific directory  
python train_hydra.py hydra.run.dir=./outputs/experiment_1

# Use different combinations
python train_hydra.py model=v1 data=fetch_pick_place training.max_epochs=50 seed=123
```

## Hyperparameter Optimization (`train_hpo_hydra.py`)

### Basic HPO
```bash
# Run scaling experiment with default settings
python train_hpo_hydra.py experiment=scaling_experiment

# Run with custom number of trials
python train_hpo_hydra.py experiment=scaling_experiment hpo.n_trials=200

# Run comparison experiment
python train_hpo_hydra.py experiment_type=comparison hpo.n_trials=100
```

### HPO with Overrides
```bash
# Custom base version for scaling
python train_hpo_hydra.py experiment=scaling_experiment model=v0 hpo.n_trials=50

# With timeout and custom study name
python train_hpo_hydra.py experiment=scaling_experiment hpo.n_trials=200 hpo.timeout=7200 hpo.study_name=my_scaling_study

# Use different dataset
python train_hpo_hydra.py experiment=scaling_experiment data=fetch_pick_place

# Custom training settings for HPO
python train_hpo_hydra.py experiment=scaling_experiment training.max_epochs=30
```

## Configuration Composition

### Creating Custom Experiments
You can create your own experiment configurations by combining different components:

```yaml
# conf/experiment/my_custom_experiment.yaml
# @package _global_
defaults:
  - model: v2              # Use v2 model
  - data: billiard_ball    # Use billiard ball dataset
  - training: default      # Use default training settings
  - hpo: null              # No HPO
  - _self_

name: my_custom_experiment
description: Custom experiment with v2 model

# Override specific model parameters
model:
  hidden_size: 256
  n_g_layers: 3
  n_r_layers: 3

# Override training parameters
training:
  max_epochs: 150
  learning_rate: 0.0005
```

### Using Custom Configurations
```bash
python train_hydra.py experiment=my_custom_experiment
```

## Key Hydra Features

### 1. **Configuration Composition**
- Mix and match components (model, data, training, etc.)
- Override any parameter from command line
- Use structured configurations with validation

### 2. **Multi-run Support** 
```bash
# Run multiple experiments with different parameters
python train_hydra.py -m model=v0,v1,v2,v3 training.learning_rate=0.001,0.01

# Run with different datasets
python train_hydra.py -m data=billiard_ball,fetch_pick_place model=v2
```

### 3. **Working Directory Management**
```bash
# Hydra automatically creates working directories
python train_hydra.py  # Creates logs/gatel0rd_experiment/2024-09-26_10-30-45/
```

### 4. **Configuration Inspection**
```bash
# Print configuration without running
python train_hydra.py --cfg job

# Print resolved configuration
python train_hydra.py --cfg resolved

# Validate configuration
python test_hydra_config.py
```

## Migration from Old API

### Before (Old API):
```bash
python train.py --config config/baseline_training.yaml
python train.py --model.version v2 --training.max_epochs 100
python train_hpo.py --experiment_type scaling --n_trials 100
```

### After (New Hydra API):
```bash
python train_hydra.py experiment=baseline_training
python train_hydra.py model=v2 training.max_epochs=100
python train_hpo_hydra.py experiment=scaling_experiment hpo.n_trials=100
```

## Benefits of Hydra Configuration

1. **Modularity**: Mix and match configuration components
2. **Type Safety**: Structured configs with validation
3. **Reproducibility**: Automatic config saving and working directories
4. **Flexibility**: Override any parameter from command line
5. **Multi-run**: Easy parameter sweeps and experiments
6. **Composition**: Build complex configurations from simple components
7. **Documentation**: Self-documenting configuration structure

## Configuration Validation

The system automatically validates:
- Data ratios sum to 1.0
- Model versions are valid (v0-v3)
- Data directories exist
- Parameter types and ranges

## Testing the Configuration

```bash
# Test configuration system
python test_hydra_config.py

# Test specific config loading
python -c "from omegaconf import OmegaConf; print(OmegaConf.load('conf/config.yaml'))"
```