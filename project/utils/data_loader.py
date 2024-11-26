# utils/data_loader.py

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import xml.etree.ElementTree as ET
# from .augmenter import DataAugmenter
from .utils import preprocess_image

class CustomDataset(Dataset):
    """객체 검출을 위한 커스텀 데이터셋"""
    
    def __init__(self,
                 data_dir: Union[str, Path],
                 transform_config: Optional[Dict] = None,
                 split: str = 'train',
                 img_size: Tuple[int, int] = (640, 640)):
        """
        Parameters:
            data_dir: 데이터셋 디렉토리 경로
            transform_config: 데이터 증강 설정
            split: 'train', 'val', 'test' 중 하나
            img_size: 입력 이미지 크기 (width, height)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        
        # 데이터 증강
        self.augmenter = DataAugmenter(transform_config) if transform_config else None
        
        # 데이터 리스트 로드
        self.data_list = self._load_data_list()
        
        # 클래스 정보 로드
        self.class_dict = self._load_class_dict()
        
    def _load_data_list(self) -> List[Dict]:
        """
        데이터 리스트 로드
        반환 형식: [{'image': image_path, 'annotation': annotation_path}, ...]
        """
        data_list = []
        
        # 이미지 디렉토리
        img_dir = self.data_dir / 'images' / self.split
        # 어노테이션 디렉토리
        ann_dir = self.data_dir / 'annotations' / self.split
        
        # 지원하는 이미지 확장자
        img_extensions = ['.jpg', '.jpeg', '.png']
        
        for img_path in img_dir.iterdir():
            if img_path.suffix.lower() in img_extensions:
                # XML 형식의 어노테이션 파일 경로
                xml_path = ann_dir / f"{img_path.stem}.xml"
                
                if xml_path.exists():
                    data_list.append({
                        'image': str(img_path),
                        'annotation': str(xml_path)
                    })
                
        return data_list
    
    def _load_class_dict(self) -> Dict[str, int]:
        """클래스 이름과 인덱스 매핑 정보 로드"""
        class_file = self.data_dir / 'classes.json'
        if not class_file.exists():
            raise FileNotFoundError(f"Class mapping file not found: {class_file}")
            
        with open(class_file, 'r') as f:
            class_dict = json.load(f)
            
        return class_dict
    
    def _parse_annotation(self, ann_path: str) -> Dict[str, np.ndarray]:
        """XML 형식의 어노테이션 파일 파싱"""
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        # 이미지 크기 정보
        size = root.find('size')
        img_width = float(size.find('width').text)
        img_height = float(size.find('height').text)
        
        boxes = []
        labels = []
        
        # 각 객체에 대한 바운딩 박스와 클래스 정보 추출
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_dict:
                continue
                
            label = self.class_dict[class_name]
            
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 박스 좌표 정규화 (0~1 범위로)
            boxes.append([
                xmin / img_width,
                ymin / img_height,
                xmax / img_width,
                ymax / img_height
            ])
            labels.append(label)
            
        # numpy 배열로 변환
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        return {
            'boxes': boxes,
            'labels': labels
        }
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        데이터셋에서 한 아이템 로드

        Returns:
            image: 전처리된 이미지 텐서 [3, H, W]
            target: 어노테이션 정보
                - boxes: 바운딩 박스 좌표 [N, 4] (normalized)
                - labels: 클래스 레이블 [N]
        """
        # 데이터 경로
        data_info = self.data_list[idx]
        
        # 이미지 로드
        image = cv2.imread(data_info['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 어노테이션 로드
        target = self._parse_annotation(data_info['annotation'])
        
        # 데이터 증강 적용 (학습 시에만)
        if self.split == 'train' and self.augmenter is not None:
            image, target = self.augmenter(image, target)
        
        # 이미지 전처리
        image = preprocess_image(image, self.img_size)
        
        # numpy -> torch 변환
        image = torch.from_numpy(image)
        target['boxes'] = torch.from_numpy(target['boxes'])
        target['labels'] = torch.from_numpy(target['labels'])
        
        return image, target

def create_dataloader(
    data_dir: Union[str, Path],
    batch_size: int,
    split: str = 'train',
    transform_config: Optional[Dict] = None,
    num_workers: int = 4,
    shuffle: bool = None
) -> torch.utils.data.DataLoader:
    """
    데이터 로더 생성 헬퍼 함수

    Parameters:
        data_dir: 데이터셋 디렉토리 경로
        batch_size: 배치 크기
        split: 'train', 'val', 'test' 중 하나
        transform_config: 데이터 증강 설정
        num_workers: 데이터 로딩에 사용할 워커 수
        shuffle: 데이터 셔플 여부 (기본값: train일 때만 True)

    Returns:
        DataLoader 객체
    """
    # Dataset 인스턴스 생성
    dataset = CustomDataset(
        data_dir=data_dir,
        transform_config=transform_config,
        split=split
    )
    
    # shuffle 기본값 설정
    if shuffle is None:
        shuffle = (split == 'train')
    
    # DataLoader 생성
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader

def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """
    배치 데이터 처리를 위한 collate 함수
    객체 검출에서는 이미지마다 객체 수가 다르므로 특별한 처리 필요
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # 이미지는 스택으로 쌓기
    images = torch.stack(images, dim=0)
    
    return images, targets



# imageNet 쓸 경우
# utils/data_loader.py

# import torch
# from torch.utils.data import Dataset
# import cv2
# import numpy as np
# from pathlib import Path
# from typing import Dict, List, Tuple, Optional, Union
# import json
# import xml.etree.ElementTree as ET
# from PIL import Image
# import os

# class ImageNetDataset(Dataset):
#     """ImageNet 데이터셋을 위한 커스텀 데이터셋"""
    
#     def __init__(self,
#                  root_dir: Union[str, Path],
#                  transform_config: Optional[Dict] = None,
#                  split: str = 'train',
#                  img_size: Tuple[int, int] = (640, 640)):
#         """
#         Parameters:
#             root_dir: ImageNet 데이터셋 루트 디렉토리
#             transform_config: 데이터 증강 설정
#             split: 'train' 또는 'val'
#             img_size: 입력 이미지 크기
#         """
#         self.root_dir = Path(root_dir)
#         self.split = split
#         self.img_size = img_size
#         self.transform_config = transform_config
        
#         # ImageNet 구조:
#         # root_dir/
#         #   ├── ILSVRC2012_img_train/
#         #   │   ├── n01440764/
#         #   │   ├── n01443537/
#         #   │   └── ...
#         #   ├── ILSVRC2012_img_val/
#         #   ├── ILSVRC2012_bbox_train_v2/
#         #   └── ILSVRC2012_bbox_val_v3/
        
#         # 데이터 경로 설정
#         self.img_dir = self.root_dir / f'ILSVRC2012_img_{split}'
#         self.bbox_dir = self.root_dir / f'ILSVRC2012_bbox_{split}_v2'
        
#         # 클래스 정보 로드
#         self.classes, self.class_to_idx = self._load_classes()
        
#         # 데이터 리스트 생성
#         self.samples = self._make_dataset()
        
#         # 데이터 증강
#         self.augmenter = DataAugmenter(transform_config) if transform_config else None

#     def _load_classes(self) -> Tuple[List[str], Dict[str, int]]:
#         """ImageNet 클래스 정보 로드"""
#         # meta.json 파일에서 클래스 정보 로드
#         meta_file = self.root_dir / 'meta.json'
#         if meta_file.exists():
#             with open(meta_file, 'r') as f:
#                 meta_data = json.load(f)
#                 classes = list(meta_data['wnid_to_classes'].keys())
#                 class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
#         else:
#             # 디렉토리에서 직접 클래스 리스트 생성
#             classes = sorted([d.name for d in self.img_dir.iterdir() if d.is_dir()])
#             class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
#         return classes, class_to_idx

#     def _make_dataset(self) -> List[Dict]:
#         """데이터셋 리스트 생성"""
#         samples = []
        
#         # 각 클래스 디렉토리 순회
#         for class_dir in self.img_dir.iterdir():
#             if not class_dir.is_dir():
#                 continue
                
#             class_name = class_dir.name
#             class_idx = self.class_to_idx[class_name]
            
#             # 이미지 파일 찾기
#             for img_path in class_dir.glob('*.JPEG'):
#                 # 바운딩 박스 XML 파일 경로
#                 bbox_path = self.bbox_dir / class_name / f"{img_path.stem}.xml"
                
#                 samples.append({
#                     'image_path': str(img_path),
#                     'bbox_path': str(bbox_path) if bbox_path.exists() else None,
#                     'class_idx': class_idx,
#                     'class_name': class_name
#                 })
        
#         return samples

#     def _parse_bbox_xml(self, xml_path: str) -> Dict[str, np.ndarray]:
#         """ImageNet 바운딩 박스 XML 파일 파싱"""
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
        
#         # 이미지 크기
#         size = root.find('size')
#         width = float(size.find('width').text)
#         height = float(size.find('height').text)
        
#         boxes = []
#         labels = []
        
#         # 객체 정보 추출
#         for obj in root.findall('object'):
#             class_name = obj.find('name').text
#             bbox = obj.find('bndbox')
            
#             # 박스 좌표 추출 및 정규화
#             xmin = float(bbox.find('xmin').text) / width
#             ymin = float(bbox.find('ymin').text) / height
#             xmax = float(bbox.find('xmax').text) / width
#             ymax = float(bbox.find('ymax').text) / height
            
#             boxes.append([xmin, ymin, xmax, ymax])
#             labels.append(self.class_to_idx[class_name])
        
#         # numpy 배열로 변환
#         boxes = np.array(boxes, dtype=np.float32)
#         labels = np.array(labels, dtype=np.int64)
        
#         return {
#             'boxes': boxes,
#             'labels': labels
#         }

#     def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
#         """데이터셋에서 한 아이템 로드"""
#         sample = self.samples[idx]
        
#         # 이미지 로드
#         image = cv2.imread(sample['image_path'])
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 바운딩 박스 정보 로드
#         if sample['bbox_path']:
#             target = self._parse_bbox_xml(sample['bbox_path'])
#         else:
#             # 바운딩 박스 정보가 없는 경우
#             target = {
#                 'boxes': np.zeros((0, 4), dtype=np.float32),
#                 'labels': np.array([sample['class_idx']], dtype=np.int64)
#             }
        
#         # 데이터 증강 적용
#         if self.split == 'train' and self.augmenter is not None:
#             image, target = self.augmenter(image, target)
        
#         # 이미지 크기 조정
#         image = cv2.resize(image, self.img_size)
#         image = image.transpose(2, 0, 1)  # HWC -> CHW
#         image = image / 255.0  # 정규화
        
#         # numpy -> torch 변환
#         image = torch.from_numpy(image).float()
#         target = {k: torch.from_numpy(v) for k, v in target.items()}
        
#         return image, target

#     def __len__(self) -> int:
#         return len(self.samples)

# # 데이터 로더 생성 함수
# def create_imagenet_dataloader(
#     root_dir: Union[str, Path],
#     batch_size: int,
#     split: str = 'train',
#     transform_config: Optional[Dict] = None,
#     num_workers: int = 4,
#     shuffle: bool = None
# ) -> torch.utils.data.DataLoader:
#     """ImageNet 데이터 로더 생성"""
#     dataset = ImageNetDataset(
#         root_dir=root_dir,
#         transform_config=transform_config,
#         split=split
#     )
    
#     if shuffle is None:
#         shuffle = (split == 'train')
    
#     return torch.utils.data.DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         num_workers=num_workers,
#         collate_fn=collate_fn,
#         pin_memory=True
#     )

# def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
#     """배치 데이터 처리"""
#     images = []
#     targets = []
    
#     for image, target in batch:
#         images.append(image)
#         targets.append(target)
    
#     images = torch.stack(images, dim=0)
    
#     return images, targets