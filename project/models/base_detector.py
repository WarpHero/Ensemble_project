from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, List, Dict, Optional
import numpy as np

class BaseDetector(nn.Module, ABC):
    """
    Object Detection 모델들의 기본 클래스
    YOLO와 Faster R-CNN의 공통 인터페이스를 정의
    """
    def __init__(self, config: Dict):
        """
        Parameters:
            config: 모델 설정을 담은 dictionary
        """
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = config['data']['num_classes']
        
    @abstractmethod
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        모델의 순전파 연산을 수행

        Parameters:
            images: 배치 이미지 텐서 [B, C, H, W]

        Returns:
            predictions: 예측 결과를 담은 dictionary
                - boxes: 바운딩 박스 좌표 [x1, y1, x2, y2]
                - scores: 각 박스의 confidence score
                - labels: 각 박스의 클래스 레이블
        """
        pass

    @abstractmethod
    def predict(self, 
                images: torch.Tensor,
                conf_threshold: Optional[float] = None,
                nms_threshold: Optional[float] = None
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        이미지에서 객체를 검출

        Parameters:
            images: 입력 이미지 텐서 [B, C, H, W]
            conf_threshold: confidence threshold (None이면 config 값 사용)
            nms_threshold: NMS threshold (None이면 config 값 사용)

        Returns:
            boxes: 검출된 바운딩 박스들 [N, 4]
            scores: 각 박스의 confidence scores [N]
            labels: 각 박스의 클래스 레이블 [N]
        """
        pass

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        입력 이미지 전처리

        Parameters:
            images: 입력 이미지 텐서 [B, C, H, W]

        Returns:
            preprocessed_images: 전처리된 이미지 텐서
        """
        # 이미지 정규화
        if not isinstance(images, torch.Tensor):
            images = torch.from_numpy(images)
        
        images = images.to(self.device).float() / 255.0
        
        # 이미지 크기 조정이 필요한 경우
        if hasattr(self, 'input_size'):
            images = torch.nn.functional.interpolate(
                images, 
                size=self.input_size,
                mode='bilinear',
                align_corners=False
            )
            
        return images

    def postprocess(self, 
                    predictions: Dict[str, torch.Tensor],
                    conf_threshold: float,
                    nms_threshold: float
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        모델 출력을 후처리하여 최종 검출 결과 생성

        Parameters:
            predictions: 모델의 원시 예측값
            conf_threshold: confidence threshold
            nms_threshold: NMS threshold

        Returns:
            processed_boxes: 후처리된 바운딩 박스들
            processed_scores: 후처리된 confidence scores
            processed_labels: 후처리된 클래스 레이블
        """
        # Confidence threshold 적용
        mask = predictions['scores'] > conf_threshold
        boxes = predictions['boxes'][mask]
        scores = predictions['scores'][mask]
        labels = predictions['labels'][mask]

        # Non-maximum suppression 적용
        keep = self.nms(boxes, scores, nms_threshold)
        
        return boxes[keep], scores[keep], labels[keep]

    def nms(self, 
            boxes: torch.Tensor,
            scores: torch.Tensor,
            threshold: float
           ) -> torch.Tensor:
        """
        Non-Maximum Suppression 수행

        Parameters:
            boxes: 바운딩 박스 좌표들 [N, 4]
            scores: confidence scores [N]
            threshold: NMS IoU threshold

        Returns:
            keep: 유지할 박스들의 인덱스
        """
        return torch.ops.torchvision.nms(boxes, scores, threshold)

    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        두 바운딩 박스 간의 IoU(Intersection over Union) 계산

        Parameters:
            box1: 첫 번째 바운딩 박스 [N, 4]
            box2: 두 번째 바운딩 박스 [M, 4]

        Returns:
            iou: IoU 값 [N, M]
        """
        # 박스 좌표 추출
        x1 = torch.max(box1[:, None, 0], box2[:, 0])  # [N,M]
        y1 = torch.max(box1[:, None, 1], box2[:, 1])
        x2 = torch.min(box1[:, None, 2], box2[:, 2])
        y2 = torch.min(box1[:, None, 3], box2[:, 3])

        # 교집합 영역 계산
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)

        # 각 박스의 면적 계산
        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]

        # 합집합 영역 계산
        union = area1[:, None] + area2 - intersection

        return intersection / (union + 1e-6)

    def load_weights(self, weights_path: str) -> None:
        """
        사전 학습된 가중치 로드

        Parameters:
            weights_path: 가중치 파일 경로
        """
        state_dict = torch.load(weights_path, map_location=self.device)
        self.load_state_dict(state_dict)

    def save_weights(self, save_path: str) -> None:
        """
        모델 가중치 저장

        Parameters:
            save_path: 저장할 경로
        """
        torch.save(self.state_dict(), save_path)

    @abstractmethod
    def get_loss(self, 
                 predictions: Dict[str, torch.Tensor],
                 targets: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """
        학습에 사용될 손실 함수 계산

        Parameters:
            predictions: 모델의 예측값
            targets: 실제 정답값

        Returns:
            losses: 각 손실 값들을 담은 dictionary
        """
        pass

    def train_step(self, 
                   images: torch.Tensor,
                   targets: Dict[str, torch.Tensor]
                  ) -> Dict[str, torch.Tensor]:
        """
        한 배치에 대한 학습 스텝 수행

        Parameters:
            images: 입력 이미지 배치
            targets: 정답값

        Returns:
            losses: 계산된 손실값들
        """
        # 순전파
        predictions = self(images)
        
        # 손실 계산
        losses = self.get_loss(predictions, targets)
        
        return losses

    def validate_step(self, 
                      images: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        검증 단계에서의 추론 수행

        Parameters:
            images: 입력 이미지 배치

        Returns:
            boxes: 검출된 바운딩 박스들
            scores: confidence scores
            labels: 클래스 레이블
        """
        with torch.no_grad():
            return self.predict(images)

    def to_cpu(self, tensor: torch.Tensor) -> np.ndarray:
        """
        텐서를 CPU numpy 배열로 변환

        Parameters:
            tensor: 변환할 텐서

        Returns:
            array: 변환된 numpy 배열
        """
        return tensor.detach().cpu().numpy()