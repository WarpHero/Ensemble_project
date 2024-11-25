# utils/metrics.py

import torch
import numpy as np
from typing import List, Tuple, Dict, Union
from collections import defaultdict

def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    두 바운딩 박스 간의 IoU(Intersection over Union) 계산

    Parameters:
        box1: 첫 번째 박스 좌표 [N, 4] (x1, y1, x2, y2)
        box2: 두 번째 박스 좌표 [M, 4] (x1, y1, x2, y2)

    Returns:
        iou: IoU 값 [N, M]
    """
    # 박스를 2D로 확장
    box1 = box1.unsqueeze(1)  # [N, 1, 4]
    box2 = box2.unsqueeze(0)  # [1, M, 4]
    
    # 교집합 영역 계산
    x1 = torch.max(box1[..., 0], box2[..., 0])
    y1 = torch.max(box1[..., 1], box2[..., 1])
    x2 = torch.min(box1[..., 2], box2[..., 2])
    y2 = torch.min(box1[..., 3], box2[..., 3])
    
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    
    # 각 박스의 면적 계산
    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    
    # 합집합 영역 계산
    union = box1_area + box2_area - intersection
    
    return intersection / (union + 1e-6)  # 0으로 나누는 것 방지

def compute_precision_recall(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    pred_labels: torch.Tensor,
    true_boxes: torch.Tensor,
    true_labels: torch.Tensor,
    iou_threshold: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precision과 Recall 계산

    Parameters:
        pred_boxes: 예측된 바운딩 박스 [N, 4]
        pred_scores: 예측 confidence scores [N]
        pred_labels: 예측된 클래스 레이블 [N]
        true_boxes: 실제 바운딩 박스 [M, 4]
        true_labels: 실제 클래스 레이블 [M]
        iou_threshold: IoU 임계값

    Returns:
        precision: 정밀도
        recall: 재현율
    """
    if pred_boxes.shape[0] == 0:
        return torch.tensor(0.0), torch.tensor(0.0)
    
    # IoU 행렬 계산
    iou_matrix = compute_iou(pred_boxes, true_boxes)
    
    # 각 예측에 대해 가장 높은 IoU를 가진 실제 박스 찾기
    max_iou, matched_idx = iou_matrix.max(dim=1)
    
    # 올바른 예측 찾기 (IoU가 임계값을 넘고 클래스가 일치하는 경우)
    correct_class = pred_labels == true_labels[matched_idx]
    true_positives = (max_iou > iou_threshold) & correct_class
    
    # Precision과 Recall 계산
    tp = true_positives.sum().float()
    fp = (~true_positives).sum().float()
    fn = (true_boxes.shape[0] - tp).float()
    
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    
    return precision, recall

def compute_ap(precisions: torch.Tensor, recalls: torch.Tensor) -> float:
    """
    Average Precision 계산 (11-point interpolation)

    Parameters:
        precisions: 정밀도 값들
        recalls: 재현율 값들

    Returns:
        ap: Average Precision
    """
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        if mask.any():
            ap += precisions[mask].max() / 11.0
            
    return ap

def compute_map(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Dict[str, float]:
    """
    전체 데이터셋에 대한 mAP 계산

    Parameters:
        predictions: 모델 예측 결과 리스트
        targets: 실제 정답 리스트
        num_classes: 클래스 수
        iou_threshold: IoU 임계값

    Returns:
        metrics: 계산된 메트릭들
            - mAP: mean Average Precision
            - AP: 각 클래스별 Average Precision
    """
    # 클래스별 예측 결과 수집
    class_predictions = defaultdict(list)
    class_targets = defaultdict(list)
    
    # 예측 결과 정리
    for pred, target in zip(predictions, targets):
        for label in range(num_classes):
            mask = pred['labels'] == label
            class_predictions[label].append({
                'boxes': pred['boxes'][mask],
                'scores': pred['scores'][mask]
            })
            
            mask = target['labels'] == label
            class_targets[label].append({
                'boxes': target['boxes'][mask]
            })
    
    # 각 클래스별 AP 계산
    aps = {}
    for label in range(num_classes):
        # 해당 클래스의 모든 예측값과 실제값 수집
        all_pred_boxes = []
        all_pred_scores = []
        all_true_boxes = []
        
        for pred, target in zip(class_predictions[label], class_targets[label]):
            all_pred_boxes.append(pred['boxes'])
            all_pred_scores.append(pred['scores'])
            all_true_boxes.append(target['boxes'])
        
        if not all_pred_boxes:  # 예측이 없는 경우
            aps[label] = 0.0
            continue
            
        # 텐서 연결
        pred_boxes = torch.cat(all_pred_boxes)
        pred_scores = torch.cat(all_pred_scores)
        true_boxes = torch.cat(all_true_boxes)
        
        # confidence score 기준으로 정렬
        sort_idx = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]
        
        # Precision-Recall 곡선 계산
        precisions = []
        recalls = []
        
        for k in range(1, len(pred_boxes) + 1):
            precision, recall = compute_precision_recall(
                pred_boxes[:k],
                pred_scores[:k],
                torch.full((k,), label),
                true_boxes,
                torch.full((len(true_boxes),), label),
                iou_threshold
            )
            precisions.append(precision)
            recalls.append(recall)
        
        precisions = torch.tensor(precisions)
        recalls = torch.tensor(recalls)
        
        # AP 계산
        aps[label] = compute_ap(precisions, recalls)
    
    # mAP 계산
    mean_ap = sum(aps.values()) / len(aps)
    
    return {
        'mAP': mean_ap,
        'AP': aps
    }

class MetricLogger:
    """메트릭 기록 및 관리"""
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def update(self, metrics: Dict[str, float]):
        """새로운 메트릭 값 추가"""
        for k, v in metrics.items():
            self.metrics[k].append(v)
            
    def average(self) -> Dict[str, float]:
        """현재까지의 평균 메트릭 값 반환"""
        return {k: sum(v) / len(v) for k, v in self.metrics.items()}
    
    def get_all(self) -> Dict[str, List[float]]:
        """모든 메트릭 기록 반환"""
        return dict(self.metrics)
    
    def reset(self):
        """메트릭 기록 초기화"""
        self.metrics.clear()

def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int
) -> Dict[str, float]:
    """
    모델 성능 평가

    Parameters:
        model: 평가할 모델
        data_loader: 테스트 데이터 로더
        device: 연산 장치
        num_classes: 클래스 수

    Returns:
        metrics: 평가 결과
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, batch_targets in data_loader:
            images = images.to(device)
            outputs = model(images)
            
            predictions.extend(outputs)
            targets.extend(batch_targets)
    
    metrics = compute_map(predictions, targets, num_classes)
    
    return metrics