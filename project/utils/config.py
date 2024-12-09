# utils/config.py
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_dir):
        self.config_dir = Path(config_dir)
        self.config = self._load_all_configs()
    
    def _load_all_configs(self):
        config = {}
        for config_file in self.config_dir.glob("**/*.yaml"):
            with open(config_file) as f:
                category = config_file.parent.name
                name = config_file.stem
                if category not in config:
                    config[category] = {}
                config[category][name] = yaml.safe_load(f)
        return config
    
    def get_config(self, category, name):
        return self.config[category][name]