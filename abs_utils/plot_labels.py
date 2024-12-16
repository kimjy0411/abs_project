import numpy as np
from yolov5.utils.plots import plot_labels
import glob
import os

# 라벨 파일 디렉토리
label_dir = 'C:/Users/jykim/Document/abs_project/data/reduced/train/labels'
save_dir = 'C:/Users/jykim/Document/abs_project/runs/plots'

# 라벨 파일 로드
label_files = glob.glob(os.path.join(label_dir, '*.txt'))
if not label_files:
    raise FileNotFoundError(f"No label files found in {label_dir}")

# 라벨 데이터 로드 및 병합
labels = np.concatenate([np.loadtxt(f).reshape(-1, 5) for f in label_files])

# 저장 디렉토리 생성
os.makedirs(save_dir, exist_ok=True)

# 클래스 분포 시각화 및 저장
plot_labels(labels, save_dir=save_dir)

print(f"클래스 분포가 {save_dir}/labels.jpg 파일에 저장되었습니다.")
