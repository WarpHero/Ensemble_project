# models/yolo_detector.py
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from ultralytics import YOLO
import cv2
import numpy as np

class YOLOWithVGG16(nn.Module):
    def __init__(self, weights_path='yolov5s.pt', version='v5', device='cuda', num_classes=80, pretrained=True):
        """
        YOLO 모델 초기화 (VGG16을 백본으로 사용)
        :param weights_path: YOLO 모델 가중치 경로 (e.g. yolov5s.pt)
        :param version: YOLO 버전 (현재는 v5만 예시)
        :param device: 'cuda' or 'cpu'
        :param num_classes: 탐지할 클래스 수 (기본 80, COCO 데이터셋 기준)
        :param pretrained: VGG16의 사전 학습된 가중치를 사용할지 여부
        """
        super(YOLOWithVGG16, self).__init__()

        self.device = device
        self.weights_path = weights_path
        self.num_classes = num_classes
        self.unfreeze_epoch = unfreeze_epoch


        # VGG16 백본 초기화 (사전 학습된 가중치 사용)
        self.backbone = VGG16Backbone(pretrained=pretrained)

        # YOLO 헤드 부분을 새로 정의
        self.yolo_head = self._create_yolo_head(num_classes)

        # YOLO 모델 가중치 로드 (기본 YOLOv5 로드, 필요시 가중치를 불러옴)
        if self.version == 'v5':
            try:
                self.model = YOLO(self.weights_path)
                print("YOLO model loaded successfully")
            except Exception as e:
                print(f"Error loading YOLO model: {str(e)}")
                raise
        else:
            raise NotImplementedError(f"YOLO version {self.version} is not supported yet.")
        
        self.model.to(self.device)
        self.model.eval()  # 추론 모드로 설정

    def _create_yolo_head(self, num_classes):
        """
        YOLO의 head 부분을 정의하는 메서드
        :param num_classes: 탐지할 클래스 수
        :return: YOLO 헤드 네트워크
        """
        return nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),  # VGG16의 출력 채널(512) -> 256
            nn.ReLU(),
            nn.Conv2d(256, num_classes * 5, kernel_size=1, stride=1)  # 각 클래스에 대해 5개 요소(바운딩 박스 4개 + 신뢰도)
        )

    def forward(self, x):
        """
        VGG16을 사용하여 특징 추출 후 YOLO 헤드로 처리
        :param x: 입력 이미지
        :return: 예측 결과
        """
        # VGG16을 통한 특징 추출
        x = self.backbone(x)
        
        # YOLO 헤드로 예측 수행
        x = self.yolo_head(x)
        return x
    def freeze_backbone(self):
        """백본을 freeze합니다."""
        self.backbone.freeze_backbone()

    def unfreeze_backbone(self):
        """백본을 unfreeze합니다."""
        self.backbone.unfreeze_backbone()
    
    def _create_yolo_head(self, num_classes):
        """YOLO 헤드 생성"""
        # YOLO 헤드에 맞는 네트워크 부분을 정의
        pass
    
    def _preprocess_image(self, img):
        """
        이미지를 YOLO 모델에 맞게 전처리
        :param img: 입력 이미지 (numpy array)
        :return: 전처리된 텐서
        """
        img_resized = cv2.resize(img, (640, 640))  # YOLOv5 모델의 입력 크기 (640x640)
        img_resized = img_resized.transpose((2, 0, 1))  # (H, W, C) -> (C, H, W)
        img_resized = np.expand_dims(img_resized, axis=0)  # (1, C, H, W)
        img_resized = torch.tensor(img_resized, dtype=torch.float32) / 255.0  # [0, 255] -> [0, 1]
        return img_resized.to(self.device)

    def predict(self, img):
        """
        이미지를 입력받아 객체 탐지를 수행하고 결과를 반환
        :param img: 입력 이미지 (numpy array, BGR 포맷)
        :return: 탐지된 객체들의 bounding box, confidence score, label
        """
        # 이미지 전처리
        img_tensor = self._preprocess_image(img)

        # 추론 수행
        with torch.no_grad():
            results = self.model(img_tensor)

        # 결과에서 bounding boxes, labels, confidence scores 추출
        boxes = results.xywh[0].cpu().numpy()  # [x_center, y_center, width, height]
        scores = results.conf[0].cpu().numpy()  # confidence scores
        labels = results.names  # class labels

        return boxes, scores, labels

    def visualize_predictions(self, img, boxes, scores, labels, conf_thres=0.3):
        """
        예측된 결과를 이미지 위에 시각화
        :param img: 원본 이미지
        :param boxes: 예측된 bounding boxes
        :param scores: 예측된 confidence scores
        :param labels: 예측된 클래스 라벨
        :param conf_thres: 최소 confidence score
        :return: 시각화된 이미지
        """
        for i in range(len(boxes)):
            if scores[i] >= conf_thres:
                x_center, y_center, width, height = boxes[i]
                x1 = int((x_center - width / 2) * img.shape[1])
                y1 = int((y_center - height / 2) * img.shape[0])
                x2 = int((x_center + width / 2) * img.shape[1])
                y2 = int((y_center + height / 2) * img.shape[0])

                # 객체 이름과 confidence score 표시
                label = f"{labels[i]} {scores[i]:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return img
