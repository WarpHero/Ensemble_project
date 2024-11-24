# models/ensemble_detector.py
class EnsembleDetector:
    yolo_model = None
    faster_rcnn = None
    
    def __init__(self,
                 conf_thres=0.3,
                 iou_thres=0.5,
                 detection_weights=(0.4, 0.6),    # YOLO, Fast R-CNN
                 classification_weights=(0.3, 0.7) # YOLO, Fast R-CNN
                ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.detection_weights = detection_weights
        self.classification_weights = classification_weights
        
        # 초기화만 수행
        self._initialize_models()
    
    @classmethod
    def _initialize_models(cls):
        """모델 초기화 (필요한 경우에만)"""
        if cls.yolo_model is None:
            print("Initializing YOLOv5...")
            cls.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            
        if cls.faster_rcnn is None:
            print("Initializing Faster R-CNN with VGG16 backbone...")
            backbone = torchvision.models.vgg16(pretrained=True).features
            cls.faster_rcnn = torchvision.models.detection.FasterRCNN(
                backbone=backbone,
                num_classes=91,  # COCO 데이터셋 기준
                box_detections_per_img=500,
                box_score_thresh=0.3
            )

    @classmethod
    def save_models(cls, save_path):
        """모델 저장"""
        torch.save({
            'yolo_state_dict': cls.yolo_model.state_dict(),
            'rcnn_state_dict': cls.faster_rcnn.state_dict(),
            'detection_weights': cls.detection_weights,
            'classification_weights': cls.classification_weights
        }, save_path)
    
    @classmethod
    def load_models(cls, load_path):
        """모델 로드"""
        checkpoint = torch.load(load_path)
        cls._initialize_models()
        cls.yolo_model.load_state_dict(checkpoint['yolo_state_dict'])
        cls.faster_rcnn.load_state_dict(checkpoint['rcnn_state_dict'])