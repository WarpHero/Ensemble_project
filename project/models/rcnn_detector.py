# models/rcnn_detector.py
import torch
import torch.nn as nn
import torchvision
import torchvision.models.detection as detection
from torchvision import transforms
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
import PIL
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

        # backbone 교체 여부 확인
        use_custom_backbone = config.get('use_custom_backbone', False)
        
        if use_custom_backbone:
            # 커스텀 백본 사용
            backbone_config = config.get('backbone', {
                'type': 'vgg16',
                'pretrained': True,
                'freeze': True
            })
            # Backbone 초기화
            self.backbone = self._create_backbone(backbone_config)
            # Faster R-CNN 모델 초기화
            self.model = self._create_faster_rcnn()
        
        else:
            # 사전 학습된 Faster R-CNN 사용
            try:
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                    pretrained=True,
                    # num_classes=self.num_classes,
                    num_classes=91,
                    box_detections_per_img=500,
                    box_score_thresh=self.conf_threshold,
                    box_nms_thresh=self.nms_threshold
                )
                print("Pretrained Faster R-CNN loaded successfully")
            except Exception as e:
                print(f"Error loading Faster R-CNN model: {str(e)}")
                raise

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
        # Faster R-CNN은 내부적으로 loss를 계산하므로 모델의 loss를 그대로 반환
        return self.model(predictions, targets)

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
        nms_threshold = nms_threshold or self.nms_threshold
        
        # 검출 수행
        results = self.detect(images)
        
        # 임계값 적용
        mask = results['scores'] >= conf_threshold
        boxes = torch.from_numpy(results['boxes'][mask]).to(self.device)
        scores = torch.from_numpy(results['scores'][mask]).to(self.device)
        labels = torch.from_numpy(results['labels'][mask]).to(self.device)
        
        return boxes, scores, labels

    def _create_backbone(self, backbone_config: Dict) -> nn.Module:
        """
        백본 네트워크 생성
        Returns:
            nn.Module: 백본 네트워크
        """
        backbone_config = {
            'pretrained': self.config.get('pretrained', True),
            'freeze': self.config.get('freeze_backbone', True)
        }
        return VGG16Backbone(backbone_config)

    def _create_faster_rcnn(self) -> nn.Module:
        """
        Faster R-CNN 모델 생성
        Returns:
            nn.Module: Faster R-CNN 모델
        """
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.rpn import AnchorGenerator

        # RPN 설정
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )

        # ROI Pooling 설정
        roi_pooler = torch.nn.ModuleDict({
            'box': torch.nn.modules.pooling.AdaptiveAvgPool2d(output_size=7)
        })

        return FasterRCNN(
            backbone=self.backbone,
            num_classes=self.num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler['box'],
            min_size=800,
            max_size=1333,
            box_score_thresh=self.conf_threshold,
            box_nms_thresh=self.nms_threshold,
            box_detections_per_img=100
        )

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