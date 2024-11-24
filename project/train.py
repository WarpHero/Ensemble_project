# train.py
def train_ensemble_model():
    # Colab 환경 설정
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Colab")
    
    # 데이터 로드
    train_loader, val_loader = load_data()
    
    # 모델 초기화
    model = EnsembleDetector(
        detection_weights=(0.4, 0.6),
        classification_weights=(0.3, 0.7)
    )
    
    # 학습 수행
    trainer = ModelTrainer(model, train_loader, val_loader)
    trainer.train(epochs=50)
    
    # 성능 평가
    evaluator = ModelEvaluator(model, val_loader)
    results = evaluator.evaluate()
    
    # 모델 저장
    model.save_models('ensemble_model.pth')

if __name__ == '__main__':
    train_ensemble_model()