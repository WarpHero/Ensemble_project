# evaluate.py

import torch
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any
import logging
import json
from datetime import datetime

from models.ensemble_detector import EnsembleDetector
from utils.data_loader import create_imagenet_dataloader
from utils.metrics import evaluate_model, MetricLogger
from utils.visualizer import Visualizer

class ModelEvaluator:
    def __init__(self, config_path: str, checkpoint_path: str):
        """
        Parameters:
            config_path: 설정 파일 경로
            checkpoint_path: 모델 체크포인트 경로
        """
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        
        # 로깅 설정
        self.logger = self._setup_logger()
        
        # 모델 및 데이터 로더 초기화
        self.model = self._load_model()
        self.val_loader = self._create_dataloader()
        
        # 시각화 도구 초기화
        self.visualizer = Visualizer(self.config)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _setup_logger(self) -> logging.Logger:
        """로깅 설정"""
        logger = logging.getLogger('Evaluation')
        logger.setLevel(logging.INFO)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # 파일 핸들러
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def _load_model(self) -> torch.nn.Module:
        """체크포인트에서 모델 로드"""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            
            # 모델 초기화
            model = EnsembleDetector(
                config=checkpoint.get('config', self.config),
                detection_weights=self.config['ensemble']['detection_weights'],
                classification_weights=self.config['ensemble']['classification_weights']
            ).to(self.device)
            
            # 가중치 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.logger.info(f"Model loaded from {self.checkpoint_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
            
    def _create_dataloader(self) -> torch.utils.data.DataLoader:
        """검증 데이터 로더 생성"""
        return create_imagenet_dataloader(
            self.config['data']['val_path'],
            batch_size=self.config['evaluation']['batch_size'],
            split='val',
            num_workers=self.config['device']['num_workers']
        )
        
    def evaluate(self) -> Dict[str, float]:
        """모델 평가 수행"""
        try:
            self.logger.info("Starting evaluation...")
            
            # 메트릭 계산
            metrics = evaluate_model(
                model=self.model,
                data_loader=self.val_loader,
                device=self.device,
                num_classes=self.config['data']['num_classes']
            )
            
            # 결과 로깅
            self.logger.info("Evaluation Results:")
            self.logger.info(f"mAP: {metrics['mAP']:.4f}")
            
            # 클래스별 AP 출력
            for class_id, ap in metrics['AP'].items():
                class_name = self.config['data']['class_names'][class_id]
                self.logger.info(f"Class {class_name} (ID: {class_id}): AP = {ap:.4f}")
            
            # 결과 저장
            self._save_results(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise
            
    def _save_results(self, metrics: Dict[str, float]):
        """평가 결과 저장"""
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'metrics': metrics,
            'model_checkpoint': str(self.checkpoint_path)
        }
        
        save_path = results_dir / f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        self.logger.info(f"Results saved to {save_path}")
        
    def visualize_results(self, num_samples: int = 10):
        """평가 결과 시각화"""
        try:
            self.logger.info(f"Visualizing {num_samples} sample predictions...")
            
            results_dir = Path('results/visualizations')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 샘플 이미지에 대한 예측 결과 시각화
            with torch.no_grad():
                for i, (images, targets) in enumerate(self.val_loader):
                    if i >= num_samples:
                        break
                        
                    images = images.to(self.device)
                    predictions = self.model(images)
                    
                    # 배치의 각 이미지에 대해
                    for j, (image, pred) in enumerate(zip(images, predictions)):
                        result_image = self.visualizer.draw_predictions(
                            image.cpu().numpy().transpose(1, 2, 0),
                            pred
                        )
                        
                        save_path = results_dir / f'sample_{i}_{j}.png'
                        self.visualizer.save_image(result_image, save_path)
                        
            self.logger.info(f"Visualizations saved to {results_dir}")
            
        except Exception as e:
            self.logger.error(f"Error during visualization: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--num-vis', type=int, default=10,
                      help='Number of samples to visualize')
    args = parser.parse_args()
    
    # 평가 수행
    evaluator = ModelEvaluator(args.config, args.checkpoint)
    metrics = evaluator.evaluate()
    
    # 결과 시각화
    if args.num_vis > 0:
        evaluator.visualize_results(args.num_vis)

if __name__ == '__main__':
    main()