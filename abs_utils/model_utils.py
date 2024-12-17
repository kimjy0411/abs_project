from ultralytics import YOLO

def load_yolo_model():
    """
    YOLO 모델 불러오기
    """
    model_path = "models/best_backup.pt"  # 모델 경로
    model = YOLO(model_path)
    return model
