# utils/augmenter.py
import cv2
import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union

class DataAugmenter:
    def __init__(self, config: Optional[Dict] = None):
        """
        데이터 증강 설정 초기화
        Args:
            config: 증강 설정
                - flip_prob: 좌우 뒤집기 확률
                - brightness: 밝기 조정 범위
                - contrast: 대비 조정 범위
                - saturation: 채도 조정 범위
                - random_crop: 랜덤 크롭 설정
        """
        self.config = config or {}
        self.flip_prob = self.config.get('flip_prob', 0.5)
        self.brightness = self.config.get('brightness', 0.2)
        self.contrast = self.config.get('contrast', 0.2)
        self.saturation = self.config.get('saturation', 0.2)
        
    def __call__(self, 
                 image: np.ndarray, 
                 target: Dict) -> Tuple[np.ndarray, Dict]:
        """
        이미지와 바운딩 박스에 증강 적용
        Args:
            image: 입력 이미지 (H, W, C)
            target: 바운딩 박스 등 타겟 정보
        Returns:
            image: 증강된 이미지
            target: 수정된 타겟
        """
        # 좌우 뒤집기
        if np.random.rand() < self.flip_prob:
            image, target = self._horizontal_flip(image, target)
        
        # 컬러 지터링
        image = self._color_jitter(image)
        
        return image, target
    
    def _horizontal_flip(self, 
                        image: np.ndarray, 
                        target: Dict) -> Tuple[np.ndarray, Dict]:
        """수평 뒤집기"""
        image = cv2.flip(image, 1)  # 1은 좌우 뒤집기
        
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes'].clone()
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]  # x 좌표 뒤집기
            target['boxes'] = boxes
            
        return image, target
    
    def _color_jitter(self, image: np.ndarray) -> np.ndarray:
        """컬러 지터링"""
        # 밝기 조정
        brightness_factor = np.random.uniform(1-self.brightness, 1+self.brightness)
        image = image * brightness_factor
        
        # 대비 조정
        contrast_factor = np.random.uniform(1-self.contrast, 1+self.contrast)
        mean = image.mean(axis=(0, 1), keepdims=True)
        image = (image - mean) * contrast_factor + mean
        
        # 채도 조정
        saturation_factor = np.random.uniform(1-self.saturation, 1+self.saturation)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = np.expand_dims(gray, -1)
        image = image * saturation_factor + gray * (1 - saturation_factor)
        
        # 값 범위 클리핑
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image



# 예시
# augmenter_config = {
#     'flip_prob': 0.5,
#     'brightness': 0.2,
#     'contrast': 0.2,
#     'saturation': 0.2
# }
