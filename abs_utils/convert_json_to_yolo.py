import os
import json

# 클래스 매핑 정의
class_mapping = {
    "Baseball_ball": 0,
    "Batter": 1,
    "Catcher": 2,
    "Umpire1": 3,
    "Home_base": 4
}

# 이미지 크기 정의
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# JSON → YOLO 변환 함수
def convert_json_to_yolo(json_path, output_dir):
    """JSON 파일을 YOLO 형식으로 변환하여 저장."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    yolo_lines = []
    for annotation in data.get("annotation", []):
        label = annotation["box"]["label"]
        if label not in class_mapping:
            continue

        # 클래스 ID 가져오기
        class_id = class_mapping[label]

        # 위치 정보 가져오기
        location = annotation["box"]["location"][0]
        x = location["x"]
        y = location["y"]
        width = location["width"]
        height = location["height"]

        # YOLO 형식으로 변환
        x_center = (x + width / 2) / IMAGE_WIDTH
        y_center = (y + height / 2) / IMAGE_HEIGHT
        rel_width = width / IMAGE_WIDTH
        rel_height = height / IMAGE_HEIGHT

        # YOLO 형식 데이터 생성
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {rel_width:.6f} {rel_height:.6f}")

    # YOLO 형식 파일 저장
    output_file = os.path.join(output_dir, os.path.basename(json_path).replace('.json', '.txt'))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(yolo_lines))

# 변환 함수 실행
def process_all_json_files(input_dir, output_dir):
    """모든 JSON 파일을 YOLO 형식으로 변환."""
    os.makedirs(output_dir, exist_ok=True)
    for json_file in os.listdir(input_dir):
        if json_file.endswith('.json'):
            convert_json_to_yolo(os.path.join(input_dir, json_file), output_dir)

# 입력과 출력 디렉토리 설정
process_all_json_files("C:/Users/jykim/Document/abs_project/data/raw/train/labels", "C:/Users/jykim/Document/abs_project/data/processed/train/labels")  # Train JSON → YOLO 변환
process_all_json_files("C:/Users/jykim/Document/abs_project/data/raw/val/labels", "C:/Users/jykim/Document/abs_project/data/processed/val/labels")      # Validation JSON → YOLO 변환

print("모든 JSON 파일이 YOLO 형식으로 변환되었습니다.")
