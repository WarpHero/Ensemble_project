import torch
import numpy as np
import random
from .ensemble_detector import EnsembleDetector
from .yolo_detector import YOLOWithVGG16 # YOLODetector
from .rcnn_detector import FasterRCNNDetector

# GPU 사용 가능 여부 확인
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 공통으로 사용할 함수 정의
def get_device():
    """장치 정보를 반환 (cuda, cpu 등)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')  # Apple Silicon GPU
    else:
        return torch.device('cpu')

def set_random_seed(seed=42):
    """랜덤 시드를 설정하여 실험 재현성 보장"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 여러 GPU에서 시드 설정

__all__ = ['EnsembleDetector', 'YOLOWithVGG16', 'FasterRCNNDetector']
