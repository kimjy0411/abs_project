import cv2
import numpy as np
from sort import Sort  # SORT 라이브러리
from ultralytics import YOLO  # YOLO 모델 로드

# SORT 객체 초기화
tracker = Sort()

def is_strike(ball_position, strike_zone):
    """
    공의 위치가 스트라이크 존에 있는지 확인합니다.
    """
    x, y = ball_position
    x_min, y_min, x_max, y_max = strike_zone
    return x_min <= x <= x_max and y_min <= y <= y_max

def calculate_strike_zone(batter_bbox, height_ratio=(0.3, 0.7)):
    """
    타자의 바운딩 박스를 기반으로 스트라이크 존을 계산합니다.
    """
    x_min, y_min, x_max, y_max = batter_bbox
    height = y_max - y_min
    strike_zone_top = int(y_min + height * height_ratio[0])
    strike_zone_bottom = int(y_min + height * height_ratio[1])
    return (x_min, strike_zone_top, x_max, strike_zone_bottom)

def process_frame_with_tracking(frame, model):
    """
    프레임을 처리하고 YOLO 모델과 SORT를 사용해 객체를 추적합니다.
    """
    results = model(frame)  # YOLO 탐지 결과
    detections = []
    strike_zone = None

    # YOLO 탐지 결과를 SORT 입력 형식으로 변환
    for result in results.pandas().xyxy[0].itertuples():
        if result.name == "Batter":
            # 타자 바운딩 박스 기반으로 스트라이크 존 생성
            batter_bbox = (int(result.xmin), int(result.ymin), int(result.xmax), int(result.ymax))
            strike_zone = calculate_strike_zone(batter_bbox)
        elif result.name == "Baseball_ball":  # 공 탐지
            x1, y1, x2, y2, conf = int(result.xmin), int(result.ymin), int(result.xmax), int(result.ymax), result.conf
            detections.append([x1, y1, x2, y2, conf])

    # SORT 알고리즘으로 객체 추적
    trackers = tracker.update(np.array(detections))

    # 프레임에 결과 표시
    for track in trackers:
        x1, y1, x2, y2, track_id = map(int, track)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2  # 공의 중심점

        # 스트라이크/볼 판정
        if strike_zone:
            is_strike_result = is_strike((center_x, center_y), strike_zone)
            label = "Strike" if is_strike_result else "Ball"
            color = (0, 255, 0) if is_strike_result else (0, 0, 255)
        else:
            label = "Tracking"
            color = (255, 255, 0)

        # 공 위치 및 라벨 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {track_id} {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 스트라이크 존 표시
        if strike_zone:
            cv2.rectangle(frame, (strike_zone[0], strike_zone[1]), (strike_zone[2], strike_zone[3]), (255, 255, 255), 2)

    return frame

def process_video(input_path, output_path, model):
    """
    동영상을 프레임 단위로 처리하고 출력 동영상을 저장합니다.
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        frame = process_frame_with_tracking(frame, model)
        out.write(frame)

    cap.release()
    out.release()
