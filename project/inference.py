from models.ensemble_detector import EnsembleDetector
from utils.visualizer import Visualizer

class Inference:
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.visualizer = Visualizer()
        
    def predict(self, image):
        # 예측 수행
        predictions = self.model.detect(image)
        
        # 결과 시각화
        result_image = self.visualizer.draw_boxes(image, predictions)
        
        return predictions, result_image
        
    def export_model(self, format: str = 'onnx'):
        # 모델 내보내기
        pass

def main():
    # 추론 예시
    inference = Inference('model.pth')
    image = load_image('test.jpg')
    predictions, result = inference.predict(image)
    save_result(result, 'output.jpg')

if __name__ == "__main__":
    main()