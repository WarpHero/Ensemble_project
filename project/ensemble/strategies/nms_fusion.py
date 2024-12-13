# ensemble/strategies/nms_fusion.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Union

class NMSFusion:
    """
    NMS 기반 앙상블 전략
    여러 detector의 예측 결과를 NMS를 사용하여 융합
    """
    def __init__(self, config: Dict):
        """
        Args:
            config: NMS 관련 설정
                - iou_threshold: NMS IoU 임계값
                - score_threshold: 점수 임계값
        """
        self.config = config
        self.iou_threshold = config.get('iou_threshold', 0.5)
        self.score_threshold = config.get('score_threshold', 0.3)
        
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
        # 모든 예측 결과를 하나로 합치기
        all_boxes = []
        all_scores = []
        all_labels = []
        
        for pred in predictions:
            all_boxes.append(pred['boxes'])
            all_scores.append(pred['scores'])
            all_labels.append(pred['labels'])
            
        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # 클래스별로 NMS 적용
        keep_indices = []
        unique_labels = torch.unique(labels)
        
        for label in unique_labels:
            class_mask = labels == label
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            # NMS 적용
            keep = self._nms(
                boxes=class_boxes,
                scores=class_scores,
                iou_threshold=self.iou_threshold
            )
            
            keep_indices.append(class_mask.nonzero()[keep])
            
        # 최종 결과 생성
        keep_indices = torch.cat(keep_indices)
        
        return {
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'labels': labels[keep_indices]
        }
    
    def _nms(self, 
             boxes: torch.Tensor,
             scores: torch.Tensor,
             iou_threshold: float) -> torch.Tensor:
        """
        Non-Maximum Suppression 수행
        Args:
            boxes: (N, 4) 바운딩 박스
            scores: (N,) 신뢰도 점수
            iou_threshold: IoU 임계값
        Returns:
            torch.Tensor: 유지할 박스들의 인덱스
        """
        if len(boxes) == 0:
            return torch.zeros(0, dtype=torch.long)
            
        # 박스 좌표
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # 박스 면적 계산
        areas = (x2 - x1) * (y2 - y1)
        
        # 점수에 따라 박스 정렬
        _, order = scores.sort(descending=True)
        
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
                
            # 가장 높은 점수의 박스 선택
            i = order[0]
            keep.append(i)
            
            # 나머지 박스들과의 IoU 계산
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            # IoU가 임계값보다 작은 박스들만 유지
            ids = (ovr <= iou_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
            
        return torch.tensor(keep)