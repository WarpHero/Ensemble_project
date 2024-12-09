# models/detector_factory.py
class DetectorFactory:
    @staticmethod
    def create_detector(detector_type, config):
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

# models/yolo_detector.py
class BaseYOLODetector(BaseDetector):
    """YOLO 모델들의 공통 기능을 구현하는 기본 클래스"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    @abstractmethod
    def load_model(self):
        pass
        
    @abstractmethod
    def preprocess(self, images):
        pass

class YOLOV8Detector(BaseYOLODetector):
    def __init__(self, config):
        super().__init__(config)
        self.model = self.load_model()
    
    def load_model(self):
        from ultralytics import YOLO
        model = YOLO(self.config['weights'])
        return model
    
    def detect(self, images):
        results = self.model(images)
        return self._process_results(results)