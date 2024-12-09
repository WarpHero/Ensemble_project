# utils/path_manager.py
import os
import yaml
from pathlib import Path

class PathManager:
    def __init__(self):
        self.is_colab = self._check_colab()
        self.paths = self._load_path_config()
        
    def _check_colab(self) -> bool:
        """Colab 환경인지 확인"""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    def _load_path_config(self) -> dict:
        """경로 설정 파일 로드"""
        config_path = Path(__file__).parent.parent / 'path_config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Colab이면 colab 설정을, 아니면 default 설정을 사용
        return config['colab'] if self.is_colab else config['default']
    
    def get_path(self, key: str) -> Path:
        """경로 반환"""
        return Path(self.paths[key])
    
    def ensure_paths(self):
        """필요한 디렉토리 생성"""
        for path in self.paths.values():
            Path(path).mkdir(parents=True, exist_ok=True)