# utils/data_loader.py
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from utils.path_manager import PathManager

# ImageNet 사용하는 경우
# import json
# import xml.etree.ElementTree as ET
# from PIL import Image

class COCODataset(Dataset):
    """COCO 데이터셋을 위한 커스텀 데이터셋"""
    
    def __init__(self, config: Dict, is_train: bool = True):
        """
        Parameters:
            config: 데이터셋 설정
            is_train: 학습/검증 모드 구분
        """
        self.config = config
        self.path_manager = PathManager()
        self.is_train = is_train
        
        # 이미지 크기 설정
        self.img_size = config.get('img_size', (640, 640))
        
        # 경로 설정
        if is_train:
            self.img_dir = Path(self.path_manager.get_path('train_path'))
            anno_file = Path(self.path_manager.get_path('annotation_path')) / "instances_train2017.json"
        else:
            self.img_dir = Path(self.path_manager.get_path('val_path'))
            anno_file = Path(self.path_manager.get_path('annotation_path')) / "instances_val2017.json"
        
        # COCO API 초기화
        self.coco = COCO(anno_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # 클래스 정보
        self.num_classes = config['dataset']['num_classes']
        
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        데이터셋에서 한 아이템 로드
        Returns:
            image: 전처리된 이미지 텐서 [3, H, W]
            target: 어노테이션 정보
                - boxes: 바운딩 박스 좌표 [N, 4] (normalized)
                - labels: 클래스 레이블 [N]
                - image_id: 이미지 ID
        """
        # 이미지 ID로 이미지와 어노테이션 로드
        img_id = self.ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)
        
        # 이미지 로드
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info['file_name']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 원본 이미지 크기
        orig_h, orig_w = image.shape[:2]
        
        # 박스와 라벨 준비
        boxes = []
        labels = []
        
        for ann in annotations:
            x, y, w, h = ann['bbox']
            # COCO 포맷(x,y,w,h)을 normalizaed (x1,y1,x2,y2)로 변환
            x1 = x / orig_w
            y1 = y / orig_h
            x2 = (x + w) / orig_w
            y2 = (y + h) / orig_h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])
        
        # 이미지 전처리
        image = cv2.resize(image, self.img_size)
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = image / 255.0  # normalize to [0, 1]
        
        # numpy -> torch 변환
        image = torch.from_numpy(image).float()
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }
        
        return image, target

def create_data_loader(config: Dict, is_train: bool = True) -> DataLoader:
    """
    데이터 로더 생성
    Args:
        config: 설정 딕셔너리
        is_train: 학습용/검증용 구분
    """
    dataset = COCODataset(config, is_train)
    
    batch_size = config['training']['batch_size'] if is_train else 1
    shuffle = is_train
    num_workers = config['device'].get('num_workers', 4)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[torch.Tensor, List[Dict]]:
    """배치 데이터 처리를 위한 collate 함수"""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    
    return images, targets


# imageNet 쓸 경우
# utils/data_loader.py

# class ImageNetDataset(Dataset):
#     """ImageNet 데이터셋을 위한 커스텀 데이터셋"""
    
#     def __init__(self, config: Dict, is_train: bool = True):
#         """
#         Parameters:
#             config: 설정 딕셔너리
#             is_train: 학습/검증 모드 구분
#         """
#         self.config = config
#         self.path_manager = PathManager()
#         self.is_train = is_train
        
#         # 이미지 크기 설정
#         self.img_size = config.get('img_size', (640, 640))
        
#         # 경로 설정
#         root_dir = self.path_manager.get_path('data_root')
#         self.split = 'train' if is_train else 'val'
#         self.img_dir = Path(root_dir) / f'ILSVRC2012_img_{self.split}'
#         self.bbox_dir = Path(root_dir) / f'ILSVRC2012_bbox_{self.split}_v2'
        
#         # 클래스 정보 로드
#         self.num_classes = config['dataset']['num_classes']
#         self.classes, self.class_to_idx = self._load_classes()
        
#         # 데이터 리스트 생성
#         self.samples = self._make_dataset()

#     def _load_classes(self) -> Tuple[List[str], Dict[str, int]]:
#         """ImageNet 클래스 정보 로드"""
#         meta_file = Path(self.path_manager.get_path('data_root')) / 'meta.json'
#         if meta_file.exists():
#             with open(meta_file, 'r') as f:
#                 meta_data = json.load(f)
#                 classes = list(meta_data['wnid_to_classes'].keys())
#                 class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
#         else:
#             classes = sorted([d.name for d in self.img_dir.iterdir() if d.is_dir()])
#             class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
#         return classes, class_to_idx

#     def _make_dataset(self) -> List[Dict]:
#         """데이터셋 리스트 생성"""
#         samples = []
#         for class_dir in self.img_dir.iterdir():
#             if not class_dir.is_dir():
#                 continue
                
#             class_name = class_dir.name
#             class_idx = self.class_to_idx[class_name]
            
#             for img_path in class_dir.glob('*.JPEG'):
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
        
#         size = root.find('size')
#         width = float(size.find('width').text)
#         height = float(size.find('height').text)
        
#         boxes = []
#         labels = []
        
#         for obj in root.findall('object'):
#             class_name = obj.find('name').text
#             bbox = obj.find('bndbox')
            
#             xmin = float(bbox.find('xmin').text) / width
#             ymin = float(bbox.find('ymin').text) / height
#             xmax = float(bbox.find('xmax').text) / width
#             ymax = float(bbox.find('ymax').text) / height
            
#             boxes.append([xmin, ymin, xmax, ymax])
#             labels.append(self.class_to_idx[class_name])
        
#         return {
#             'boxes': np.array(boxes, dtype=np.float32),
#             'labels': np.array(labels, dtype=np.int64)
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
#             target = {
#                 'boxes': np.zeros((0, 4), dtype=np.float32),
#                 'labels': np.array([sample['class_idx']], dtype=np.int64)
#             }
        
#         # 이미지 전처리
#         image = cv2.resize(image, self.img_size)
#         image = image.transpose(2, 0, 1)  # HWC -> CHW
#         image = image / 255.0  # 정규화
        
#         # numpy -> torch 변환
#         image = torch.from_numpy(image).float()
#         target = {k: torch.from_numpy(v) for k, v in target.items()}
#         target['image_id'] = torch.tensor([idx])  # 이미지 ID 추가
        
#         return image, target

#     def __len__(self) -> int:
#         return len(self.samples)

# def create_data_loader(config: Dict, is_train: bool = True) -> DataLoader:
#     """
#     데이터 로더 생성
#     Args:
#         config: 설정 딕셔너리
#         is_train: 학습용/검증용 구분
#     """
#     dataset = ImageNetDataset(config, is_train)
    
#     batch_size = config['training']['batch_size'] if is_train else 1
#     shuffle = is_train
#     num_workers = config['device'].get('num_workers', 4)
    
#     return DataLoader(
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