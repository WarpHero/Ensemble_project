# models/rcnn_detector.py

import torch
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# import detectron2
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog

class FasterRCNNDetector:
    def __init__(self, device='cuda'):
        """
        Faster R-CNN 모델 초기화 (PyTorch의 torchvision 모델 사용)
        :param device: 'cuda' or 'cpu'
        """
        self.device = device

        # 모델 초기화
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()  # 평가 모드로 설정
        self.model.to(self.device)  # 모델을 지정된 장치로 이동

        # 이미지 변환 (transforms)
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 이미지를 Tensor로 변환
        ])

    def predict(self, img):
        """
        이미지를 입력받아 객체 탐지를 수행하고 결과를 반환
        :param img: 입력 이미지 (PIL Image 또는 numpy array, RGB 포맷)
        :return: 탐지된 객체들의 bounding box, confidence score, label
        """
        # 이미지를 Tensor로 변환하고 배치 차원 추가
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # 예측 수행
        with torch.no_grad():
            outputs = self.model(img_tensor)

        # 결과 추출
        boxes = outputs[0]["boxes"].cpu().numpy()  # bounding boxes
        scores = outputs[0]["scores"].cpu().numpy()  # confidence scores
        labels = outputs[0]["labels"].cpu().numpy()  # 클래스 라벨

        return boxes, scores, labels

    def visualize_predictions(self, img, boxes, scores, labels, conf_thres=0.3):
        """
        예측된 결과를 이미지 위에 시각화
        :param img: 원본 이미지 (PIL Image)
        :param boxes: 예측된 bounding boxes
        :param scores: 예측된 confidence scores
        :param labels: 예측된 클래스 라벨
        :param conf_thres: 최소 confidence score
        :return: 시각화된 이미지
        """
        # 결과 필터링: 신뢰도가 설정된 threshold 이상인 객체만 시각화
        indices = np.where(scores > conf_thres)[0]

        # 시각화
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img)

        # bounding boxes 그리기
        for idx in indices:
            box = boxes[idx]
            label = labels[idx]
            score = scores[idx]

            # bounding box 그리기
            x1, y1, x2, y2 = box
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # 라벨과 신뢰도 표시
            ax.text(x1, y1, f'{label}: {score:.2f}', color='red', fontsize=12)

        plt.axis('off')
        plt.show()

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