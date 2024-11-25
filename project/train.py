# train.py
import yaml
import torch
import logging
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.ensemble_detector import EnsembleDetector
from models.yolo_detector import YOLODetector
from models.rcnn_detector import RCNNDetector
from utils.data_loader import CustomDataset
from utils.metrics import evaluate_model
from utils.logger import setup_logger
from utils.trainer import Trainer

def load_config(config_path: str) -> dict:
    """설정 파일 로드"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_device() -> torch.device:
    """CUDA 사용 가능 여부 확인 및 device 설정"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True  # 성능 최적화
    else:
        raise RuntimeError("CUDA not available in Colab")
    return device

def create_dataloaders(config: dict) -> tuple:
    """데이터 로더 생성"""
    train_dataset = CustomDataset(
        data_dir=config['data']['train_path'],
        transforms=config['data']['augmentation'],
        is_training=True
    )
    
    val_dataset = CustomDataset(
        data_dir=config['data']['val_path'],
        transforms=None,  # 검증 시에는 augmentation 미적용
        is_training=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['device']['num_workers'],
        pin_memory=config['device']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['device']['num_workers'],
        pin_memory=config['device']['pin_memory']
    )
    
    return train_loader, val_loader

def create_models(config: dict, device: torch.device) -> EnsembleDetector:
    """모델 생성"""
    # YOLO 모델 초기화
    # yolo_model = YOLODetector(config).to(device)
    yolo_model = YOLOWithVGG16(weights_path=config['model']['yolo_weights'], device=device).to(device)

    # Faster R-CNN 모델 초기화
    rcnn_model = RCNNDetector(config).to(device)
    
    # Ensemble 모델 생성
    ensemble_model = EnsembleDetector(
        yolo_model=yolo_model,
        rcnn_model=rcnn_model,
        detection_weights=config['ensemble']['detection_weights'],
        classification_weights=config['ensemble']['classification_weights']
    ).to(device)
    
    return ensemble_model

def create_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    """옵티마이저 생성"""
    # VGG16 백본과 헤드를 구분해서 optimizer를 설정
    params_to_update = []
    
    # VGG16 백본은 requires_grad=False로 설정하여 학습하지 않게 할 수 있음
    for param in model.backbone.parameters():
        if param.requires_grad:
            params_to_update.append(param)
    
    # YOLO 헤드 부분은 학습
    for param in model.yolo_head.parameters():
        params_to_update.append(param)
    
    optimizer = AdamW(params_to_update, lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    return optimizer

    # return AdamW(
    #     model.parameters(),
    #     lr=config['training']['learning_rate'],
    #     weight_decay=config['training']['weight_decay']
    # )

def create_scheduler(optimizer: torch.optim.Optimizer, config: dict, num_training_steps: int) -> torch.optim.lr_scheduler._LRScheduler:
    """학습률 스케줄러 생성"""
    if config['training']['scheduler']['type'] == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps,
            eta_min=config['training']['scheduler']['min_lr']
        )
    # 다른 스케줄러 타입 추가 가능
    return None

def main():
    # 설정 파일 로드
    config = load_config('configs/config.yaml')
    
    # 로깅 설정
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(config['logging']['log_dir']) / current_time
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir / 'training.log')
    
    # 디바이스 설정
    device = setup_device()
    logger.info(f"Using device: {device}")
    
    # 데이터 로더 생성
    train_loader, val_loader = create_dataloaders(config)
    logger.info(f"Train dataset size: {len(train_loader.dataset)}")
    logger.info(f"Val dataset size: {len(val_loader.dataset)}")
    
    # 모델 생성
    model = create_models(config, device)
    logger.info("Model created successfully")
    
    # 옵티마이저와 스케줄러 생성
    optimizer = create_optimizer(model, config)
    num_training_steps = len(train_loader) * config['training']['epochs']
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Trainer 초기화
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger,
        log_dir=log_dir
    )
    
    # 학습 시작
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        # 중간 결과 저장
        trainer.save_checkpoint(log_dir / 'interrupted_checkpoint.pth')
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise
    
    # 최종 모델 저장
    final_model_path = log_dir / 'final_model.pth'
    trainer.save_checkpoint(final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # 최종 성능 평가
    results = trainer.evaluate()
    logger.info("Final evaluation results:")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()