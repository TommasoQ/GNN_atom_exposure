"""
Configuration management utilities
"""
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """Configuration class for managing hyperparameters and settings."""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.

        Args:
            config_dict (dict): Configuration dictionary
        """
        self._config = config_dict
        self._update_attributes()

    def _update_attributes(self):
        """Update object attributes from config dictionary."""
        for key, value in self._config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        """Allow dictionary-style access."""
        return self._config[key]

    def __setitem__(self, key, value):
        """Allow dictionary-style assignment."""
        self._config[key] = value
        setattr(self, key, value)

    def get(self, key, default=None):
        """Get value with default fallback."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return self._config.copy()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            yaml_path (str): Path to YAML file

        Returns:
            Config: Configuration object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(config_dict)

    def save_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.

        Args:
            yaml_path (str): Path to save YAML file
        """
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def __repr__(self):
        """String representation."""
        return f"Config({self._config})"


# Default configuration
DEFAULT_CONFIG = {
    'data': {
        'root': 'dataset/',
        'batch_size': 8,
        'num_workers': 4,
        'train_split': 0.8,
        'val_split': 0.1,
        'test_split': 0.1
    },
    'model': {
        'model_type': 'gnn',
        'in_channels': 80,
        'hidden_channels': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'conv_type': 'gcn'
    },
    'training': {
        'num_epochs': 100,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'scheduler': 'reduce_on_plateau',
        'patience': 10,
        'min_lr': 1e-6
    },
    'experiment': {
        'name': 'atom_exposure_gnn',
        'checkpoint_dir': 'experiments/checkpoints',
        'log_dir': 'experiments/logs',
        'save_best': True,
        'seed': 42
    }
}


def get_default_config() -> Config:
    """
    Get default configuration.

    Returns:
        Config: Default configuration object
    """
    return Config(DEFAULT_CONFIG)


if __name__ == '__main__':
    # Test configuration
    config = get_default_config()
    print(config)

    # Access nested values
    print(f"\nBatch size: {config.data.batch_size}")
    print(f"Hidden channels: {config.model.hidden_channels}")
    print(f"Learning rate: {config.training.learning_rate}")

    # Save to YAML
    config.save_yaml('configs/default_config.yaml')
    print("\nSaved default configuration to configs/default_config.yaml")
