import os
import shutil

def move_images(source_dir, target_dir):
    """
    이미지 파일을 source_dir에서 target_dir로 이동합니다.
    """
    os.makedirs(target_dir, exist_ok=True)  # 타겟 디렉토리 생성

    for file in os.listdir(source_dir):
        if file.endswith('.jpg') or file.endswith('.png'):  # 이미지 파일만 이동
            source_path = os.path.join(source_dir, file)
            target_path = os.path.join(target_dir, file)
            shutil.move(source_path, target_path)  # 파일 이동
            print(f"Moved: {source_path} -> {target_path}")

# Train 이미지 이동
move_images("C:/Users/jykim/Document/abs/data/raw/train/images", "C:/Users/jykim/Document/abs/data/processed/train/images")

# Validation 이미지 이동
move_images("C:/Users/jykim/Document/abs/data/raw/val/images", "C:/Users/jykim/Document/abs/data/processed/val/images")

print("모든 이미지 파일이 이동되었습니다.")
