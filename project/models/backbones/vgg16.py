import torch
import torchvision.models as models
from torch import nn

class VGG16Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG16Backbone, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        self.features = vgg16.features  # VGG16의 특징 추출 부분만 사용
        self.avgpool = vgg16.avgpool    # 풀링 레이어
        self.fc = vgg16.classifier      # 마지막 분류 부분은 사용하지 않음

    def forward(self, x):
        x = self.features(x)   # 특징 추출
        x = self.avgpool(x)    # 풀링
        return x
