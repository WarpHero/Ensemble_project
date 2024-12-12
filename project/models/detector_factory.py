# models/detector_factory.py
from typing import Dict
from .base_detector import BaseDetector
from .yolo_detector import YOLOV8Detector
from .rcnn_detector import FasterRCNNDetector

class DetectorFactory:
    @staticmethod
    def create_detector(detector_type: str, config):
        if detector_type.startswith('yolo'):
            version = config['version']
            if version == 'v8':
                return YOLOV8Detector(config)
            elif version == 'v5':
                return YOLOV5Detector(config)
            # 새로운 버전이 나오면 여기에 추가
            else:
                raise ValueError(f"Unsupported YOLO version: {version}")
        elif detector_type == 'faster_rcnn':
            return FasterRCNNDetector(config)
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
