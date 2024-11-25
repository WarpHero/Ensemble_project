# models/rcnn_detector.py

import torch
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

class FasterRCNNDetector:
    def __init__(self, weights_path='COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml', device='cuda'):
        """
        Faster R-CNN 모델 초기화 (Pretrained weights 사용)
        :param weights_path: Detectron2에서 제공하는 사전 훈련된 모델 구성 파일 경로
        :param device: 'cuda' or 'cpu'
        """
        self.device = device

        # Detectron2 설정
        cfg = get_cfg()
        # 모델 아키텍처와 가중치를 로드
        cfg.merge_from_file(model_zoo.get_config_file(weights_path))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_path)
        cfg.MODEL.DEVICE = self.device  # 사용할 장치 설정

        # 추론기(predictor) 초기화
        self.predictor = DefaultPredictor(cfg)

    def predict(self, img):
        """
        이미지를 입력받아 객체 탐지를 수행하고 결과를 반환
        :param img: 입력 이미지 (numpy array, BGR 포맷)
        :return: 탐지된 객체들의 bounding box, confidence score, label
        """
        # 예측 수행
        outputs = self.predictor(img)
        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()  # bounding boxes
        scores = outputs["instances"].scores.cpu().numpy()  # confidence scores
        labels = outputs["instances"].pred_classes.cpu().numpy()  # 클래스 라벨

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
        # Visualizer 초기화
        metadata = MetadataCatalog.get("coco_2017_val")  # COCO 데이터셋 메타데이터
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)

        # 신뢰도가 일정 기준 이상인 객체만 시각화
        instances = outputs["instances"]
        instances = instances[instances.scores > conf_thres]

        vis_output = visualizer.draw_instance_predictions(instances.to("cpu"))
        return vis_output.get_image()  # 시각화된 이미지 반환
