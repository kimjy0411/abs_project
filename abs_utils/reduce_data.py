import os
import random
import shutil

# TRAIN 데이터 축소 함수
def reduce_train_data(input_images_dir, input_labels_dir, output_images_dir, output_labels_dir, reduction_ratio=0.2):
    """
    TRAIN 데이터를 비율에 맞게 축소하여 reduced 폴더에 저장.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # 이미지와 라벨 파일 가져오기
    image_files = sorted([f for f in os.listdir(input_images_dir) if f.endswith(('.jpg', '.png'))])
    label_files = sorted([f for f in os.listdir(input_labels_dir) if f.endswith('.txt')])  # 이미 YOLO 형식

    print(f"이미지 파일 수: {len(image_files)}")
    print(f"라벨 파일 수: {len(label_files)}")

    # 파일 이름 매칭
    matched_files = [
        (img, img.replace('.jpg', '.txt').replace('.png', '.txt')) 
        for img in image_files 
        if img.replace('.jpg', '.txt').replace('.png', '.txt') in label_files
    ]

    print(f"매칭된 파일 수: {len(matched_files)}")

    # 축소된 데이터 샘플링
    sample_size = int(len(matched_files) * reduction_ratio)
    sampled_files = random.sample(matched_files, sample_size)

    # 축소 데이터를 reduced 폴더에 저장
    for img_file, lbl_file in sampled_files:
        shutil.copy(os.path.join(input_images_dir, img_file), os.path.join(output_images_dir, img_file))
        shutil.copy(os.path.join(input_labels_dir, lbl_file), os.path.join(output_labels_dir, lbl_file))

    print(f"TRAIN 데이터 축소 완료: {len(sampled_files)}개의 파일이 {output_images_dir}에 저장되었습니다.")

# 경로 설정
input_images_dir = "C:/Users/jykim/Document/abs/data/processed/train/images"
input_labels_dir = "C:/Users/jykim/Document/abs/data/processed/train/labels"
output_images_dir = "C:/Users/jykim/Document/abs/data/reduced/train/images"
output_labels_dir = "C:/Users/jykim/Document/abs/data/reduced/train/labels"

# 실행 (10% 축소)
reduce_train_data(input_images_dir, input_labels_dir, output_images_dir, output_labels_dir, reduction_ratio=0.10)
