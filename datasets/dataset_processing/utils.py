import yaml
import logging
from pathlib import Path

def load_yaml_config(config_path: str) -> dict:
    """Load dataset configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Dictionary containing configuration data.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logging(config_path: str = 'dataset_config.yaml') -> None:
    """Set up logging configuration.

    Args:
        config_file: Path to log file.
    """
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )