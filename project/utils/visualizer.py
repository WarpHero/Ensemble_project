# utils/visualizer.py

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import random

class Visualizer:
    def __init__(self, config: dict):
        """
        Parameters:
            config: 설정 파일에서 로드한 설정값들
        """
        self.config = config
        self.class_names = config['data']['class_names']
        self.colors = self._generate_colors(len(self.class_names))
        
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """각 클래스별 고유한 색상 생성"""
        random.seed(42)  # 재현성을 위한 시드 설정
        colors = []
        for _ in range(num_classes):
            color = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
            colors.append(color)
        return colors
    
    def draw_predictions(self,
                        image: np.ndarray,
                        predictions: Dict[str, Union[np.ndarray, torch.Tensor]],
                        confidence_threshold: float = 0.3
                       ) -> np.ndarray:
        """
        이미지에 검출 결과 시각화

        Parameters:
            image: 원본 이미지 (RGB)
            predictions: 모델의 예측 결과
                - boxes: 바운딩 박스 좌표 [N, 4] (x1, y1, x2, y2)
                - scores: 각 박스의 confidence score [N]
                - labels: 각 박스의 클래스 레이블 [N]
            confidence_threshold: 표시할 검출 결과의 최소 confidence score

        Returns:
            visualized_image: 시각화된 이미지
        """
        image = image.copy()
        
        # 텐서를 numpy로 변환
        if isinstance(predictions['boxes'], torch.Tensor):
            boxes = predictions['boxes'].cpu().numpy()
            scores = predictions['scores'].cpu().numpy()
            labels = predictions['labels'].cpu().numpy()
        else:
            boxes = predictions['boxes']
            scores = predictions['scores']
            labels = predictions['labels']
            
        # confidence threshold 적용
        mask = scores >= confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # 각 검출 결과 시각화
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = map(int, box)
            color = self.colors[int(label)]
            
            # 바운딩 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 레이블 텍스트
            label_text = f'{self.class_names[int(label)]} {score:.2f}'
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # 레이블 배경 그리기
            cv2.rectangle(
                image,
                (x1, y1 - text_size[1] - 4),
                (x1 + text_size[0], y1),
                color,
                -1
            )
            
            # 레이블 텍스트 그리기
            cv2.putText(
                image,
                label_text,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            
        return image
    
    def plot_training_results(self,
                            metrics: Dict[str, List[float]],
                            save_path: str = None
                           ) -> Union[Figure, None]:
        """
        학습 결과 그래프 생성

        Parameters:
            metrics: 각 에폭별 메트릭 기록
                - train_loss: 학습 손실
                - val_loss: 검증 손실
                - mAP: Mean Average Precision
                등
            save_path: 그래프를 저장할 경로 (None이면 화면에 표시)

        Returns:
            fig: matplotlib Figure 객체 (save_path가 None일 때)
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # 손실 그래프
        axes[0].plot(metrics['train_loss'], label='Train Loss')
        axes[0].plot(metrics['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # mAP 그래프
        axes[1].plot(metrics['mAP'], label='mAP')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('mAP')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return None
        return fig
    
    def visualize_batch(self,
                       images: torch.Tensor,
                       targets: Dict[str, torch.Tensor] = None,
                       predictions: Dict[str, torch.Tensor] = None,
                       max_images: int = 16
                      ) -> Figure:
        """
        배치 이미지와 정답/예측 결과 시각화

        Parameters:
            images: 배치 이미지 텐서 [B, C, H, W]
            targets: 정답 데이터 (있는 경우)
            predictions: 모델 예측 결과 (있는 경우)
            max_images: 시각화할 최대 이미지 개수

        Returns:
            fig: matplotlib Figure 객체
        """
        batch_size = min(images.size(0), max_images)
        fig_size = min(batch_size, 4)
        fig, axes = plt.subplots(
            (batch_size + fig_size - 1) // fig_size,
            fig_size,
            figsize=(15, 15)
        )
        
        if batch_size == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx in range(batch_size):
            # 이미지 변환 (텐서 -> numpy)
            image = images[idx].cpu().numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
            
            # 정답 바운딩 박스 (있는 경우)
            if targets is not None:
                boxes = targets['boxes'][targets['batch_idx'] == idx]
                labels = targets['labels'][targets['batch_idx'] == idx]
                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = box.cpu().numpy()
                    rect = patches.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        linewidth=2,
                        edgecolor='g',
                        facecolor='none'
                    )
                    axes[idx].add_patch(rect)
            
            # 예측 바운딩 박스 (있는 경우)
            if predictions is not None:
                boxes = predictions['boxes'][predictions['batch_idx'] == idx]
                scores = predictions['scores'][predictions['batch_idx'] == idx]
                labels = predictions['labels'][predictions['batch_idx'] == idx]
                
                for box, score, label in zip(boxes, scores, labels):
                    if score > 0.3:  # confidence threshold
                        x1, y1, x2, y2 = box.cpu().numpy()
                        rect = patches.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            linewidth=2,
                            edgecolor='r',
                            facecolor='none'
                        )
                        axes[idx].add_patch(rect)
            
            axes[idx].imshow(image)
            axes[idx].axis('off')
            
        # 사용하지 않는 subplot 제거
        for idx in range(batch_size, len(axes)):
            fig.delaxes(axes[idx])
            
        plt.tight_layout()
        return fig