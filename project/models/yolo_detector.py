# models/yolo_detector.py
import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, Tuple, List, Optional

from .base_detector import BaseDetector
from .backbones.vgg16 import VGG16Backbone

class YOLOV8Detector(BaseDetector):
    def __init__(self, config: Dict):
        """
        YOLO 디텍터 초기화
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)
        
        # 설정 불러오기
        self.weights_path = config.get('weights', 'yolov8s.pt')
        self.num_classes = config.get('num_classes', 80)
        self.conf_threshold = config.get('conf_threshold', 0.3)
        self.iou_threshold = config.get('iou_threshold', 0.5)
        
        # VGG16 백본 초기화
        backbone_config = {
            'pretrained': config.get('pretrained', True),
            'freeze': config.get('freeze_backbone', True)
        }
        self.backbone = VGG16Backbone(backbone_config)
        
        # YOLO 모델 초기화
        try:
            self.model = YOLO(self.weights_path)
            print(f"YOLO model loaded successfully from {self.weights_path}")
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            raise

        # YOLO 헤드 초기화
        self.yolo_head = self._create_yolo_head()
        
        self.to(self.device)

    def _create_yolo_head(self) -> nn.Module:
        """
        YOLO 헤드 네트워크 생성
        Returns:
            nn.Module: YOLO 헤드 네트워크
        """
        channels = self.backbone.get_output_channels()
        last_stage_channels = channels['stage5']  # VGG16의 마지막 스테이지 출력 채널

        return nn.Sequential(
            nn.Conv2d(last_stage_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, self.num_classes * (5 + self.num_classes), kernel_size=1)
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        순전파 수행
        Args:
            images: 입력 이미지 배치 (B, C, H, W)
        Returns:
            torch.Tensor: YOLO 출력
        """
        features = self.backbone(images)
        out = self.yolo_head(features['stage5'])
    
        return {
            'features': features,
            'predictions': out
        }


    def detect(self, images: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        객체 검출 수행
        Args:
            images: 입력 이미지 배치
        Returns:
            Dict: 검출 결과 (boxes, scores, labels)
        """
        self.eval()
        with torch.no_grad():
            results = self.model(images)
            
        # 결과 후처리
        boxes = []
        scores = []
        labels = []
        
        for result in results:
            boxes.append(result.boxes.xyxy.cpu().numpy())
            scores.append(result.boxes.conf.cpu().numpy())
            labels.append(result.boxes.cls.cpu().numpy().astype(int))
            
        return {
            'boxes': np.concatenate(boxes) if boxes else np.array([]),
            'scores': np.concatenate(scores) if scores else np.array([]),
            'labels': np.concatenate(labels) if labels else np.array([])
        }

    def get_loss(self,
                 predictions: Dict[str, torch.Tensor],
                 targets: Dict[str, torch.Tensor]
                 ) -> Dict[str, torch.Tensor]:
        """
        학습에 사용될 손실 함수 계산
        Args:
            predictions: 모델의 예측값
            targets: 실제 정답값
        Returns:
            Dict: 각 손실 값들을 담은 dictionary
        """
        # YOLOv8은 내부적으로 loss를 계산하므로 모델의 loss를 그대로 반환
        return self.model.loss(predictions, targets)

    def predict(self, images: torch.Tensor, conf_threshold: Optional[float] = None, 
                nms_threshold: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        이미지에서 객체를 검출
        Args:
            images: 입력 이미지 텐서 [B, C, H, W]
            conf_threshold: confidence threshold
            nms_threshold: NMS threshold
        Returns:
            Tuple: (boxes, scores, labels)
                - boxes: 검출된 바운딩 박스들 [N, 4]
                - scores: 각 박스의 confidence scores [N]
                - labels: 각 박스의 클래스 레이블 [N]
        """
        # 임계값 설정
        conf_threshold = conf_threshold or self.conf_threshold
        nms_threshold = nms_threshold or self.iou_threshold
        
        # 검출 수행
        results = self.detect(images)
        
        # 임계값 적용
        mask = results['scores'] >= conf_threshold
        boxes = torch.from_numpy(results['boxes'][mask]).to(self.device)
        scores = torch.from_numpy(results['scores'][mask]).to(self.device)
        labels = torch.from_numpy(results['labels'][mask]).to(self.device)
        
        return boxes, scores, labels

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        이미지 전처리
        Args:
            image: OpenCV 이미지 (H, W, C), BGR 형식
        Returns:
            torch.Tensor: 전처리된 이미지 텐서
        """
        # YOLOv8 기본 전처리
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        img /= 255.0
        return img.unsqueeze(0).to(self.device)

    def get_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        이미지에서 특징 추출
        Args:
            images: 입력 이미지 배치
        Returns:
            Dict: 각 스테이지별 특징 맵
        """
        return self.backbone(images)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        학습 스텝 수행
        Args:
            batch: 학습 배치
        Returns:
            Dict: 손실값들
        """
        self.train()
        return self.model.train_step(batch)

    @property
    def model_type(self) -> str:
        return 'yolo'