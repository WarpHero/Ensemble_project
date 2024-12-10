# models/rcnn_detector.py
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from PIL import Image

from .base_detector import BaseDetector
from .backbones.vgg16 import VGG16Backbone

class FasterRCNNDetector(BaseDetector):
    def __init__(self, config: Dict):
        """
        Faster R-CNN 디텍터 초기화
        Args:
            config: 설정 딕셔너리
        """
        super().__init__(config)

        # 설정 불러오기
        self.num_classes = config.get('num_classes', 81)  # COCO 기준 (80 + 배경)
        self.conf_threshold = config.get('conf_threshold', 0.3)
        self.nms_threshold = config.get('nms_threshold', 0.5)
        
        # Backbone 초기화
        self.backbone = self._create_backbone()
        
        # Faster R-CNN 모델 초기화
        self.model = self._create_faster_rcnn()
        self.model.to(self.device)

        # 이미지 변환
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        순전파 수행
        Args:
            images: 입력 이미지 배치 (B, C, H, W)
        Returns:
            Dict: Faster R-CNN 출력 (boxes, scores, labels)
        """
        return self.model(images)

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
            predictions = self.model(images)
            
        # 결과 후처리
        boxes = []
        scores = []
        labels = []
        
        for pred in predictions:
            boxes.append(pred['boxes'].cpu().numpy())
            scores.append(pred['scores'].cpu().numpy())
            labels.append(pred['labels'].cpu().numpy())
            
        return {
            'boxes': np.concatenate(boxes) if boxes else np.array([]),
            'scores': np.concatenate(scores) if scores else np.array([]),
            'labels': np.concatenate(labels) if labels else np.array([])
        }

    def get_features(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        이미지에서 특징 추출
        Args:
            images: 입력 이미지 배치
        Returns:
            Dict: 각 스테이지별 특징 맵
        """
        return self.backbone(images)

    def preprocess(self, image: Union[np.ndarray, PIL.Image.Image]) -> torch.Tensor:
        """
        이미지 전처리
        Args:
            image: OpenCV 이미지 또는 PIL 이미지
        Returns:
            torch.Tensor: 전처리된 이미지 텐서
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        return self.transform(image).unsqueeze(0).to(self.device)

    @property
    def model_type(self) -> str:
        return 'faster_rcnn'

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        학습 스텝 수행
        Args:
            batch: 학습 배치
        Returns:
            Dict: 손실값들
        """
        self.train()
        images = batch['images']
        targets = batch['targets']
        
        loss_dict = self.model(images, targets)
        return {k: v.item() for k, v in loss_dict.items()}

# 사용 예시
if __name__ == '__main__':
    # 모델 초기화
    detector = FasterRCNNDetector(device='cuda' if torch.cuda.is_available() else 'cpu')

    # 테스트 이미지 로드
    image_path = 'path_to_your_image.jpg'
    img = Image.open(image_path).convert("RGB")

    # 예측 수행
    boxes, scores, labels = detector.predict(img)

    # 예측 결과 시각화
    detector.visualize_predictions(img, boxes, scores, labels, conf_thres=0.3)


"""
# VGG16 백본 사용
config = {
    'backbone': {
        'type': 'vgg16',
        'pretrained': True,
        'freeze': True
    },
    'num_classes': 91
}

# ResNet50 백본 사용
config = {
    'backbone': {
        'type': 'resnet50',
        'pretrained': True,
        'freeze': True,
        'trainable_stages': [4, 5]
    },
    'num_classes': 91
}

detector = FasterRCNNDetector(config)
"""