# utils/utils.py

def preprocess_image(image: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
    """이미지 전처리"""
    # 크기 조정
    resized = cv2.resize(image, input_size)
    # 정규화 및 채널 순서 변경 (HWC -> CHW)
    processed = resized.transpose(2, 0, 1).astype('float32') / 255.0
    return processed

def postprocess_predictions(
    predictions: Dict[str, torch.Tensor],
    original_size: Tuple[int, int],
    input_size: Tuple[int, int]
) -> Dict[str, np.ndarray]:
    """예측 결과 후처리"""
    # 입력 크기에서 원본 크기로 바운딩 박스 좌표 변환
    height_scale = original_size[0] / input_size[0]
    width_scale = original_size[1] / input_size[1]
    
    boxes = predictions['boxes'].cpu().numpy()
    boxes[:, [0, 2]] *= width_scale   # x 좌표
    boxes[:, [1, 3]] *= height_scale  # y 좌표
    
    return {
        'boxes': boxes,
        'scores': predictions['scores'].cpu().numpy(),
        'labels': predictions['labels'].cpu().numpy()
    }