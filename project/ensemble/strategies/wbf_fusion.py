# ensemble/strategies/wbf_fusion.py
import torch
import numpy as np
from typing import Dict, List, Tuple, Union

class WBFFusion:
   """
   Weighted Box Fusion 기반 앙상블 전략
   여러 detector의 예측 결과를 WBF를 사용하여 융합
   """
   def __init__(self, config: Dict):
       """
       Args:
           config: WBF 관련 설정
               - iou_threshold: IoU 임계값
               - score_threshold: 점수 임계값
               - weights: 각 detector의 가중치
       """
       self.config = config
       self.iou_threshold = config.get('iou_threshold', 0.5)
       self.score_threshold = config.get('score_threshold', 0.3)
       self.weights = config.get('weights', None)  # detector별 가중치
       
   def __call__(self, 
                predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
       """
       여러 detector의 예측 결과를 융합
       Args:
           predictions: 각 detector의 예측 결과 리스트
               각 예측은 Dict 형태:
               - boxes: (N, 4) 바운딩 박스
               - scores: (N,) 신뢰도 점수
               - labels: (N,) 클래스 레이블
       Returns:
           Dict: 융합된 예측 결과
       """
       # detector별 가중치 설정
       if self.weights is None:
           self.weights = [1.0] * len(predictions)
           
       # 모든 예측 결과를 하나로 합치기
       all_boxes = []
       all_scores = []
       all_labels = []
       all_weights = []
       
       for pred, weight in zip(predictions, self.weights):
           all_boxes.append(pred['boxes'])
           all_scores.append(pred['scores'] * weight)  # 가중치 적용
           all_labels.append(pred['labels'])
           all_weights.extend([weight] * len(pred['boxes']))
           
       boxes = torch.cat(all_boxes, dim=0)
       scores = torch.cat(all_scores, dim=0)
       labels = torch.cat(all_labels, dim=0)
       weights = torch.tensor(all_weights, device=boxes.device)
       
       # 클래스별로 WBF 적용
       final_boxes = []
       final_scores = []
       final_labels = []
       
       unique_labels = torch.unique(labels)
       for label in unique_labels:
           class_mask = labels == label
           class_boxes = boxes[class_mask]
           class_scores = scores[class_mask]
           class_weights = weights[class_mask]
           
           # WBF 적용
           fused_boxes, fused_scores = self._weighted_box_fusion(
               boxes=class_boxes,
               scores=class_scores,
               weights=class_weights
           )
           
           if len(fused_boxes) > 0:
               final_boxes.append(fused_boxes)
               final_scores.append(fused_scores)
               final_labels.extend([label] * len(fused_boxes))
               
       if not final_boxes:  # 결과가 없는 경우
           return {
               'boxes': torch.zeros((0, 4), device=boxes.device),
               'scores': torch.zeros(0, device=boxes.device),
               'labels': torch.zeros(0, dtype=torch.long, device=boxes.device)
           }
           
       final_boxes = torch.cat(final_boxes, dim=0)
       final_scores = torch.cat(final_scores, dim=0)
       final_labels = torch.tensor(final_labels, device=boxes.device)
       
       return {
           'boxes': final_boxes,
           'scores': final_scores,
           'labels': final_labels
       }
   
   def _weighted_box_fusion(self,
                          boxes: torch.Tensor,
                          scores: torch.Tensor,
                          weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
       """
       Weighted Box Fusion 수행
       Args:
           boxes: (N, 4) 바운딩 박스
           scores: (N,) 신뢰도 점수
           weights: (N,) 각 박스의 가중치
       Returns:
           Tuple[torch.Tensor, torch.Tensor]: 융합된 박스와 점수
       """
       if len(boxes) == 0:
           return torch.zeros((0, 4), device=boxes.device), torch.zeros(0, device=boxes.device)
           
       # 점수가 임계값보다 높은 박스만 선택
       mask = scores > self.score_threshold
       if not mask.any():
           return torch.zeros((0, 4), device=boxes.device), torch.zeros(0, device=boxes.device)
           
       boxes = boxes[mask]
       scores = scores[mask]
       weights = weights[mask]
       
       # 박스 클러스터링
       clusters = []
       used = set()
       
       for i in range(len(boxes)):
           if i in used:
               continue
               
           cluster = [i]
           used.add(i)
           
           for j in range(i + 1, len(boxes)):
               if j in used:
                   continue
                   
               if self._iou(boxes[i], boxes[j]) > self.iou_threshold:
                   cluster.append(j)
                   used.add(j)
                   
           clusters.append(cluster)
           
       # 각 클러스터에 대해 가중 평균 계산
       fused_boxes = []
       fused_scores = []
       
       for cluster in clusters:
           cluster_boxes = boxes[cluster]
           cluster_scores = scores[cluster]
           cluster_weights = weights[cluster]
           
           # 가중치와 점수를 결합
           combined_weights = cluster_weights * cluster_scores
           norm_weights = combined_weights / combined_weights.sum()
           
           # 박스 좌표의 가중 평균 계산
           weighted_box = (cluster_boxes * norm_weights.view(-1, 1)).sum(dim=0)
           weighted_score = cluster_scores.mean()  # 또는 max나 다른 방식 사용 가능
           
           fused_boxes.append(weighted_box)
           fused_scores.append(weighted_score)
           
       return torch.stack(fused_boxes), torch.tensor(fused_scores)
   
   def _iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
       """두 박스 간의 IoU 계산"""
       x1 = torch.max(box1[0], box2[0])
       y1 = torch.max(box1[1], box2[1])
       x2 = torch.min(box1[2], box2[2])
       y2 = torch.min(box1[3], box2[3])
       
       intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
       box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
       box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
       union = box1_area + box2_area - intersection
       
       return intersection / union