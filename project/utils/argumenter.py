# utils/augmenter.py
class DataAugmenter:
    """Simple pass-through class for compatibility"""
    def __init__(self, config=None):
        pass
        
    def __call__(self, image, target):
        return image, target