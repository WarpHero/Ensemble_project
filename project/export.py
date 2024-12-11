# export.py

import torch
from pathlib import Path
from typing import Tuple, Union, Dict
import logging

class ModelExporter:
    def __init__(self, model_path: str, config_path: str = 'configs/config.yaml'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config(config_path)
        self.model = self._load_model(model_path)
        self.logger = self._setup_logger()
        
    def export(self, 
               output_path: Union[str, Path],
               format: str = 'onnx',
               input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640)
              ) -> None:
        """
        모델을 지정된 포맷으로 변환
        Args:
            output_path: 저장 경로
            format: 변환 포맷 ('onnx', 'torchscript')
            input_shape: 입력 텐서 shape
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'onnx':
                self._export_onnx(output_path, input_shape)
            elif format.lower() == 'torchscript':
                self._export_torchscript(output_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            self.logger.info(f"Model exported successfully to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {str(e)}")
            raise
            
    def _export_onnx(self, output_path: Path, input_shape: Tuple):
        """ONNX 포맷으로 변환"""
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

    def _export_torchscript(self, output_path: Path):
        """TorchScript 포맷으로 변환"""
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(output_path)

if __name__ == "__main__":
    model_path = 'checkpoints/best_model.pth'
    exporter = ModelExporter(model_path)
    
    # ONNX로 변환
    exporter.export('exported_models/model.onnx', format='onnx')
    
    # TorchScript로 변환
    exporter.export('exported_models/model.pt', format='torchscript')