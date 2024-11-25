# inference.py
import torch
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Union, List
import logging
from PIL import Image

from models.ensemble_detector import EnsembleDetector
from utils.visualizer import Visualizer
from utils.preprocess import preprocess_image
from utils.postprocess import postprocess_predictions

class Inference:
    def __init__(self, model_path: str, config_path: str = 'configs/config.yaml'):
        """
        Parameters:
            model_path: 학습된 모델 체크포인트 경로
            config_path: 설정 파일 경로
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config(config_path)
        self.model = self._load_model(model_path)
        self.visualizer = Visualizer(self.config)
        self.logger = self._setup_logger()
        
    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger('Inference')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger
        
    def _load_model(self, model_path: str) -> EnsembleDetector:
        """모델과 가중치 로드"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 모델 초기화
            model = EnsembleDetector(
                config=checkpoint.get('config', self.config),
                detection_weights=self.config['ensemble']['detection_weights'],
                classification_weights=self.config['ensemble']['classification_weights']
            ).to(self.device)
            
            # 가중치 로드
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            self.logger.info(f"Model loaded successfully from {model_path}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
        
    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """이미지 로드 및 전처리"""
        try:
            if isinstance(image_path, str):
                image_path = Path(image_path)
                
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
                
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image: {str(e)}")
            raise
            
    @torch.no_grad()
    def predict(self, 
                image: Union[str, Path, np.ndarray],
                conf_threshold: float = None,
                nms_threshold: float = None
               ) -> Tuple[Dict, np.ndarray]:
        """
        이미지에서 객체 검출 수행
        
        Parameters:
            image: 입력 이미지 경로 또는 numpy 배열
            conf_threshold: confidence threshold (None이면 config 값 사용)
            nms_threshold: NMS threshold (None이면 config 값 사용)
            
        Returns:
            predictions: 검출 결과 (boxes, scores, labels)
            result_image: 시각화된 결과 이미지
        """
        try:
            # 이미지 로드
            if isinstance(image, (str, Path)):
                image = self.load_image(image)
            
            # 이미지 전처리
            preprocessed_image = preprocess_image(
                image,
                input_size=self.config['model']['input_size']
            )
            
            # 텐서 변환 및 디바이스 이동
            input_tensor = torch.from_numpy(preprocessed_image).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)
            
            # 추론
            predictions = self.model.predict(
                input_tensor,
                conf_threshold or self.config['model']['conf_threshold'],
                nms_threshold or self.config['model']['nms_threshold']
            )
            
            # 후처리
            processed_predictions = postprocess_predictions(
                predictions,
                original_size=image.shape[:2],
                input_size=self.config['model']['input_size']
            )
            
            # 결과 시각화
            result_image = self.visualizer.draw_predictions(
                image.copy(),
                processed_predictions,
                class_names=self.config['data']['class_names']
            )
            
            return processed_predictions, result_image
            
        except Exception as e:
            self.logger.error(f"Error during inference: {str(e)}")
            raise
            
    def predict_batch(self, 
                     image_paths: List[Union[str, Path]],
                     output_dir: Union[str, Path],
                     batch_size: int = 8
                    ) -> List[Dict]:
        """배치 추론 수행"""
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results = []
            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i:i + batch_size]
                batch_images = [self.load_image(path) for path in batch_paths]
                
                for img, path in zip(batch_images, batch_paths):
                    predictions, result_image = self.predict(img)
                    results.append(predictions)
                    
                    # 결과 저장
                    output_path = output_dir / f"result_{Path(path).stem}.jpg"
                    cv2.imwrite(str(output_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error during batch inference: {str(e)}")
            raise
        
    def export_model(self, 
                    output_path: Union[str, Path],
                    format: str = 'onnx',
                    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)
                   ) -> None:
        """
        모델을 다른 포맷으로 내보내기
        
        Parameters:
            output_path: 저장할 경로
            format: 변환 포맷 ('onnx', 'torchscript')
            input_shape: 입력 텐서 shape (batch_size, channels, height, width)
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'onnx':
                dummy_input = torch.randn(input_shape).to(self.device)
                torch.onnx.export(
                    self.model,
                    dummy_input,
                    output_path,
                    opset_version=11,
                    input_names=['input'],
                    output_names=['boxes', 'scores', 'labels'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'boxes': {0: 'batch_size'},
                        'scores': {0: 'batch_size'},
                        'labels': {0: 'batch_size'}
                    }
                )
                
            elif format.lower() == 'torchscript':
                scripted_model = torch.jit.script(self.model)
                scripted_model.save(output_path)
                
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            self.logger.info(f"Model exported successfully to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {str(e)}")
            raise

def main():
    """추론 예시"""
    try:
        # 설정
        model_path = 'checkpoints/best_model.pth'
        config_path = 'configs/config.yaml'
        image_path = 'test_images/test.jpg'
        output_path = 'results/output.jpg'
        
        # 추론기 초기화
        inference = Inference(model_path, config_path)
        
        # 단일 이미지 추론
        predictions, result_image = inference.predict(image_path)
        
        # 결과 저장
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # 결과 출력
        print("Detection Results:")
        for box, score, label in zip(
            predictions['boxes'],
            predictions['scores'],
            predictions['labels']
        ):
            print(f"Class: {label}, Score: {score:.4f}, Box: {box}")
        
        # 모델 내보내기 (선택사항)
        inference.export_model('exported_models/model.onnx', format='onnx')
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()