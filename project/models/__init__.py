# models/__init__.py
import torch
from .ensemble_detector import EnsembleDetector
from .yolo_detector import YOLODetector
from .rcnn_detector import RCNNDetector

# GPU 사용 가능 여부 확인
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 공통으로 사용할 함수 정의
def get_device():
    return DEVICE

def set_random_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

__all__ = ['EnsembleDetector', 'YOLODetector', 'RCNNDetector']

