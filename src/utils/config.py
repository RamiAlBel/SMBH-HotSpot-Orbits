import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load and validate experiment configuration from YAML file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_keys = ['experiment', 'data', 'targets', 'noise', 'model', 'training', 'split']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    
    return config


def get_repo_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent.parent
