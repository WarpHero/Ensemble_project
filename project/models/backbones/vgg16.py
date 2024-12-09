# models/backbones/vgg16.py
import torch
import torchvision.models as models
from torch import nn
from typing import Dict, Optional, Tuple

class VGG16Backbone(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super(VGG16Backbone, self).__init__()
        
        if config is None:
            config = {'pretrained': True}
            
        # VGG16 모델 로드
        vgg16 = models.vgg16(pretrained=config.get('pretrained', True))
        
        # Feature Extractor 부분만 사용
        self.features = nn.ModuleList()
        self.feature_layers = {}  # 중간 피처맵 저장을 위한 딕셔너리
        
        # VGG16 레이어를 단계별로 나누어 저장
        current_layer = []
        layer_counter = 1
        
        for layer in vgg16.features:
            current_layer.append(layer)
            if isinstance(layer, nn.MaxPool2d):
                self.feature_layers[f'stage{layer_counter}'] = nn.Sequential(*current_layer)
                current_layer = []
                layer_counter += 1
        
        # 마지막 레이어 추가
        if current_layer:
            self.feature_layers[f'stage{layer_counter}'] = nn.Sequential(*current_layer)
        
        # 모듈로 등록
        self.stages = nn.ModuleDict(self.feature_layers)
        
        if config.get('freeze', True):
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        각 스테이지의 피처맵을 반환합니다.
        
        Args:
            x: 입력 이미지 텐서 (B, C, H, W)
            
        Returns:
            Dict[str, torch.Tensor]: 각 스테이지의 피처맵을 담은 딕셔너리
        """
        features = {}
        for name, stage in self.stages.items():
            x = stage(x)
            features[name] = x
        return features

    def freeze_backbone(self, stages: Optional[Tuple[int, ...]] = None):
        """
        특정 스테이지 또는 전체 백본을 고정합니다.
        
        Args:
            stages: 고정할 스테이지 번호들의 튜플. None이면 전체 고정
        """
        if stages is None:
            # 전체 백본 고정
            for param in self.parameters():
                param.requires_grad = False
            print("All backbone stages are frozen.")
        else:
            # 특정 스테이지만 고정
            for stage_num in stages:
                stage_name = f'stage{stage_num}'
                if stage_name in self.stages:
                    for param in self.stages[stage_name].parameters():
                        param.requires_grad = False
                    print(f"Stage {stage_num} is frozen.")

    def unfreeze_backbone(self, stages: Optional[Tuple[int, ...]] = None):
        """
        특정 스테이지 또는 전체 백본을 훈련 가능하게 설정합니다.
        
        Args:
            stages: 훈련 가능하게 할 스테이지 번호들의 튜플. None이면 전체 해제
        """
        if stages is None:
            # 전체 백본 훈련 가능
            for param in self.parameters():
                param.requires_grad = True
            print("All backbone stages are unfrozen.")
        else:
            # 특정 스테이지만 훈련 가능
            for stage_num in stages:
                stage_name = f'stage{stage_num}'
                if stage_name in self.stages:
                    for param in self.stages[stage_name].parameters():
                        param.requires_grad = True
                    print(f"Stage {stage_num} is unfrozen.")
    
    def get_output_channels(self, stage: Optional[int] = None) -> Dict[str, int]:
        """
        각 스테이지의 출력 채널 수를 반환합니다.
        
        Args:
            stage: 특정 스테이지의 채널 수만 반환할 경우 해당 스테이지 번호
            
        Returns:
            Dict[str, int]: 각 스테이지별 출력 채널 수
        """
        channels = {}
        for name, stage_module in self.stages.items():
            # 마지막 Conv 레이어의 출력 채널 수 찾기
            for m in reversed(stage_module):
                if isinstance(m, nn.Conv2d):
                    channels[name] = m.out_channels
                    break
        
        if stage is not None:
            stage_name = f'stage{stage}'
            return {stage_name: channels[stage_name]}
        return channels