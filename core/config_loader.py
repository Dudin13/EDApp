import yaml
from pathlib import Path

class ConfigNode:
    """Recursively converts a dictionary into an object with attribute access."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)
                
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self):
        return f"ConfigNode({self.to_dict()})"

class ConfigLoader:
    _config = None

    @classmethod
    def get_config(cls, config_filename="config.yaml"):
        if cls._config is None:
            # Assume project root is the parent directory of 'core'
            base_dir = Path(__file__).resolve().parent.parent
            path = base_dir / config_filename
            
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found at: {path}")

            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            cls._config = ConfigNode(data)
            
        return cls._config

# Pre-load configuration so it can be imported directly as an object
config = ConfigLoader.get_config()
