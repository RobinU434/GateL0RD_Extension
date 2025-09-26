#!/usr/bin/env python3
"""
Test script to validate Hydra configuration system.

This script tests the new Hydra-based configuration without actually running training.
"""

from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_hydra_config():
    """Test that Hydra configurations load correctly."""
    
    try:
        from omegaconf import OmegaConf
        from project.utils.config_hydra import ConfigManager, ExperimentConfig
        
        print("✓ Successfully imported Hydra and OmegaConf")
        
        # Test loading structured config
        config = ExperimentConfig()
        config_dict = OmegaConf.structured(config)
        print("✓ Successfully created structured config")
        
        # Test config manager
        manager = ConfigManager()
        print("✓ Successfully created ConfigManager")
        
        # Test config validation
        manager.validate_config(config_dict)
        print("✓ Configuration validation passed")
        
        # Test directory creation
        dirs = manager.create_experiment_dirs(config_dict)
        print(f"✓ Created experiment directories: {list(dirs.keys())}")
        
        print("\n🎉 All Hydra configuration tests passed!")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install hydra-core and omegaconf:")
        print("pip install hydra-core>=1.3.0 omegaconf>=2.3.0")
        return False
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    return True


def test_config_files():
    """Test that configuration files are properly structured."""
    
    try:
        from omegaconf import OmegaConf
        
        conf_dir = Path("conf")
        if not conf_dir.exists():
            print("❌ conf/ directory not found")
            return False
        
        # Test main config
        main_config = OmegaConf.load(conf_dir / "config.yaml")
        print("✓ Successfully loaded main config")
        
        # Test model configs
        model_configs = list((conf_dir / "model").glob("*.yaml"))
        for model_config in model_configs:
            cfg = OmegaConf.load(model_config)
            print(f"✓ Successfully loaded model config: {model_config.name}")
        
        # Test data configs
        data_configs = list((conf_dir / "data").glob("*.yaml"))
        for data_config in data_configs:
            cfg = OmegaConf.load(data_config)
            print(f"✓ Successfully loaded data config: {data_config.name}")
        
        # Test training configs
        training_configs = list((conf_dir / "training").glob("*.yaml"))
        for training_config in training_configs:
            cfg = OmegaConf.load(training_config)
            print(f"✓ Successfully loaded training config: {training_config.name}")
        
        print("\n🎉 All configuration files loaded successfully!")
        
    except Exception as e:
        print(f"❌ Configuration file test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    print("Testing Hydra Configuration System")
    print("=" * 50)
    
    success = True
    
    print("\n1. Testing Hydra imports and basic functionality...")
    success &= test_hydra_config()
    
    print("\n2. Testing configuration files...")
    success &= test_config_files()
    
    if success:
        print("\n🎉 All tests passed! Hydra configuration system is ready to use.")
        print("\nUsage examples:")
        print("  python train_hydra.py")
        print("  python train_hydra.py experiment=baseline_training")
        print("  python train_hydra.py model=v2 data=fetch_pick_place")
        print("  python train_hpo_hydra.py experiment=scaling_experiment")
    else:
        print("\n❌ Some tests failed. Please check the configuration setup.")
        sys.exit(1)