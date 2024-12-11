# models/__init__.py
import torch
import numpy as np
import random
from typing import Union

from .ensemble_detector import EnsembleDetector 
from .detector_factory import DetectorFactory
from .base_detector import BaseDetector

def get_device() -> torch.device:
   """
   사용 가능한 장치 반환
   Returns:
       torch.device: cuda, mps 또는 cpu
   """
   if torch.cuda.is_available():
       return torch.device('cuda')
   elif torch.backends.mps.is_available():
       return torch.device('mps')  # Apple Silicon GPU
   else:
       return torch.device('cpu')

def set_random_seed(seed: int = 42) -> None:
   """
   실험 재현성을 위한 랜덤 시드 설정
   Args:
       seed: 랜덤 시드 값
   """
   torch.manual_seed(seed)
   np.random.seed(seed)
   random.seed(seed)
   if torch.cuda.is_available():
       torch.cuda.manual_seed(seed)
       torch.cuda.manual_seed_all(seed)
       torch.backends.cudnn.deterministic = True
       torch.backends.cudnn.benchmark = False

def create_detector(config: dict) -> Union[BaseDetector, EnsembleDetector]:
   """
   설정에 따라 detector 생성
   Args:
       config: 설정 딕셔너리
   Returns:
       BaseDetector 또는 EnsembleDetector 인스턴스
   """
   if config.get('ensemble', False):
       return EnsembleDetector(config)
   else:
       factory = DetectorFactory()
       return factory.create_detector(config['detector_type'], config)

__all__ = [
   'EnsembleDetector',
   'BaseDetector', 
   'DetectorFactory',
   'get_device',
   'set_random_seed',
   'create_detector'
]