# inference.py
class ModelDeployment:
    def __init__(self, model_path):
        self.model = EnsembleDetector.load_models(model_path)
        self.model.eval()
        
    def export_onnx(self, save_path):
        """ONNX 포맷으로 변환"""
        dummy_input = torch.randn(1, 3, 640, 640)
        torch.onnx.export(self.model, dummy_input, save_path)
    
    def optimize_for_deployment(self):
        """배포를 위한 최적화"""
        self.model = torch.jit.script(self.model)
        return self