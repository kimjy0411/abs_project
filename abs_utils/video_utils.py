import cv2
import os

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

def is_valid_batter(bbox, frame_height, threshold_ratio=0.5):
    """
    타자를 포수와 구분하기 위해 높이 비율을 기준으로 판단합니다.
    """
    _, y_min, _, y_max = bbox
    bbox_height = y_max - y_min
    return bbox_height / frame_height >= threshold_ratio

def process_frame(frame, model, frame_height):
    """
    프레임을 처리하고 YOLO 모델을 사용해 스트라이크/볼 판정을 수행합니다.
    """
    results = model(frame)
    ball_position = None
    strike_zone = None

    # 타자만 탐지 (높이 비율로 필터링)
    for _, row in results.pandas().xyxy[0].iterrows():
        if row['name'] == 'Batter':
            bbox = (int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax']))
            if is_valid_batter(bbox, frame_height):
                strike_zone = calculate_strike_zone(bbox)
        elif row['name'] == 'Baseball_ball':
            x_center = int((row['xmin'] + row['xmax']) / 2)
            y_center = int((row['ymin'] + row['ymax']) / 2)
            ball_position = (x_center, y_center)

    if ball_position is None or strike_zone is None:
        return frame, None

    # 스트라이크/볼 판정
    is_strike_result = is_strike(ball_position, strike_zone)
    label = "Strike" if is_strike_result else "Ball"
    color = (0, 255, 0) if is_strike_result else (0, 0, 255)

    # 결과 표시
    cv2.rectangle(frame, (strike_zone[0], strike_zone[1]), (strike_zone[2], strike_zone[3]), (255, 255, 255), 2)
    cv2.circle(frame, ball_position, 10, color, -1)
    cv2.putText(frame, label, (ball_position[0] - 20, ball_position[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame, is_strike_result

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

        frame, _ = process_frame(frame, model, height)
        out.write(frame)

    cap.release()
    out.release()
