# models/ensemble_detector.py
import torch
import torchvision
import numpy as np
from typing import Dict, List, Tuple, Optional
from PIL import Image
from ultralytics import YOLO

class YOLOV8Detector(BaseDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = YOLO(config['weights'])
        self.model.to(self.device)
        
    def detect(self, image: torch.Tensor) -> Dict:
        results = self.model(image)
        return self._process_results(results[0])
    
    def _process_results(self, result) -> Dict:
        return {
            'boxes': result.boxes.xyxy.cpu().numpy(),
            'scores': result.boxes.conf.cpu().numpy(),
            'labels': result.boxes.cls.cpu().numpy().astype(int)
        }
        
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

class FasterRCNNDetector(BaseDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        backbone = torchvision.models.vgg16(pretrained=True).features
        self.model = torchvision.models.detection.FasterRCNN(
            backbone=backbone,
            num_classes=91,
            box_detections_per_img=500,
            box_score_thresh=config.get('conf_threshold', 0.3)
        )
        self.model.to(self.device)
        
    def detect(self, image: torch.Tensor) -> Dict:
        self.model.eval()
        with torch.no_grad():
            predictions = self.model([image])[0]
        return predictions
        
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return torchvision.transforms.functional.to_tensor(image)

class EnsembleDetector:
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 설정값 초기화
        self.conf_thres = config.get('conf_threshold', 0.3)
        self.iou_thres = config.get('iou_threshold', 0.5)
        self.detection_weights = config.get('detection_weights', (0.4, 0.6))
        self.classification_weights = config.get('classification_weights', (0.3, 0.7))
        
        # 모델 초기화
        factory = DetectorFactory()
        self.yolo_model = factory.create_detector('yolo', config['yolo'])
        self.faster_rcnn = factory.create_detector('faster_rcnn', config['rcnn'])

    def detect(self, image: Image.Image) -> Dict:
        """객체 감지 수행"""
        # 이미지 전처리
        yolo_input = self.yolo_model.preprocess(image).to(self.device)
        rcnn_input = self.faster_rcnn.preprocess(image).to(self.device)
        
        # 예측 수행
        yolo_pred = self.yolo_model.detect(yolo_input)
        rcnn_pred = self.faster_rcnn.detect(rcnn_input)
        
        # 앙상블 예측 수행
        return self.ensemble_predictions(yolo_pred, rcnn_pred, image.size)

    def ensemble_predictions(self, yolo_pred: Dict, rcnn_pred: Dict, image_size: Tuple[int, int]) -> Dict:
        """두 모델의 예측 결과를 통합"""
        # YOLOv8 결과 처리
        yolo_boxes = yolo_pred['boxes']
        yolo_scores = yolo_pred['scores'] * self.detection_weights[0]
        yolo_labels = yolo_pred['labels']
        
        # Faster R-CNN 결과 처리
        rcnn_boxes = rcnn_pred['boxes'].cpu().numpy()
        rcnn_scores = rcnn_pred['scores'].cpu().numpy() * self.detection_weights[1]
        rcnn_labels = rcnn_pred['labels'].cpu().numpy()
        
        # 모든 예측 통합
        boxes = np.vstack([yolo_boxes, rcnn_boxes]) if len(yolo_boxes) > 0 else rcnn_boxes
        scores = np.hstack([yolo_scores, rcnn_scores]) if len(yolo_scores) > 0 else rcnn_scores
        labels = np.hstack([yolo_labels, rcnn_labels]) if len(yolo_labels) > 0 else rcnn_labels
        
        # Weighted Box Fusion 적용
        final_boxes, final_scores, final_labels = self.weighted_box_fusion(boxes, scores, labels)
        
        return {
            'boxes': final_boxes,
            'scores': final_scores,
            'labels': final_labels
        }

    def weighted_box_fusion(self, boxes, scores, labels):
        """Weighted Box Fusion 구현"""
        clusters = self.cluster_boxes(boxes)
        
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for cluster_indices in clusters:
            cluster_boxes = boxes[cluster_indices]
            cluster_scores = scores[cluster_indices]
            cluster_labels = labels[cluster_indices]
            
            # 클래스별 가중치 적용
            unique_labels = np.unique(cluster_labels)
            label_scores = {}
            
            for label in unique_labels:
                label_mask = (cluster_labels == label)
                yolo_mask = label_mask & (cluster_scores <= 1.0)
                rcnn_mask = label_mask & (cluster_scores > 1.0)
                
                # Fast R-CNN에 더 높은 가중치 부여
                yolo_confidence = np.sum(cluster_scores[yolo_mask]) * self.classification_weights[0]
                rcnn_confidence = np.sum(cluster_scores[rcnn_mask]) * self.classification_weights[1]
                
                label_scores[label] = yolo_confidence + rcnn_confidence
            
            # 최종 클래스 및 박스 결정
            final_label = max(label_scores.items(), key=lambda x: x[1])[0]
            weights = cluster_scores / np.sum(cluster_scores)
            weighted_box = np.sum(cluster_boxes * weights[:, np.newaxis], axis=0)
            
            final_boxes.append(weighted_box)
            final_scores.append(np.mean(cluster_scores))
            final_labels.append(final_label)
        
        return np.array(final_boxes), np.array(final_scores), np.array(final_labels)

    def cluster_boxes(self, boxes):
        """IoU 기반 박스 클러스터링"""
        clusters = []
        used = set()
        
        for i in range(len(boxes)):
            if i in used:
                continue
                
            current_cluster = [i]
            used.add(i)
            
            for j in range(i + 1, len(boxes)):
                if j in used:
                    continue
                    
                if self.compute_iou(boxes[i], boxes[j]) >= self.iou_thres:
                    current_cluster.append(j)
                    used.add(j)
            
            clusters.append(current_cluster)
        
        return clusters

    @staticmethod
    def compute_iou(box1, box2):
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (box1_area + box2_area - intersection)

    def preprocess_for_yolo(self, image: Image.Image) -> torch.Tensor:
        """YOLO용 이미지 전처리"""
        # 이미지를 torch Tensor로 변환 후 배치 차원 추가
        img = np.array(image)  # PIL 이미지 -> NumPy array
        img = torch.tensor(img).float()  # NumPy -> Torch tensor
        img = img.permute(2, 0, 1) / 255.0  # HWC -> CHW, normalize to [0, 1]
        return img.unsqueeze(0).to(self.device)

    def preprocess_for_rcnn(self, image: Image.Image) -> torch.Tensor:
        """Fast R-CNN용 이미지 전처리"""
        return torchvision.transforms.functional.to_tensor(image).to(self.device)
