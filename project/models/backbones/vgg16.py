import torch
import torchvision.models as models
from torch import nn

class VGG16Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16Backbone, self).__init__()
        
        # VGG16 모델을 불러와서, 특징 추출 부분만 사용
        vgg16 = models.vgg16(pretrained=pretrained)
        self.features = vgg16.features  # VGG16의 특징 추출 부분
        self.avgpool = vgg16.avgpool    # 풀링 레이어
        self.fc = vgg16.classifier      # 마지막 분류 부분은 사용하지 않음

        # 모든 파라미터를 freeze 상태로 초기화
        self.freeze_backbone()

    def forward(self, x):
        """특징 추출 과정"""
        x = self.features(x)   # 특징 추출
        x = self.avgpool(x)    # 풀링
        return x

    def freeze_backbone(self):
        """백본의 모든 파라미터를 freeze합니다."""
        for param in self.parameters():
            param.requires_grad = False
        print("Backbone is frozen.")

    def unfreeze_backbone(self):
        """백본의 모든 파라미터를 unfreeze합니다."""
        for param in self.parameters():
            param.requires_grad = True
        print("Backbone is unfrozen.")
