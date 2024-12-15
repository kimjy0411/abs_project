import cv2
import os

def is_strike(ball_position, strike_zone):
    """
    공의 위치가 스트라이크 존에 있는지 확인합니다.
    """
    x, y = ball_position
    x_min, y_min, x_max, y_max = strike_zone
    return x_min <= x <= x_max and y_min <= y <= y_max

def process_frame(frame, model):
    """
    프레임을 처리하고 YOLO 모델을 사용해 스트라이크/볼 판정을 수행합니다.
    """
    height, width, _ = frame.shape
    # 스트라이크 존 설정
    strike_zone = (int(width * 0.3), int(height * 0.4), int(width * 0.7), int(height * 0.6))

    # YOLO 모델로 공 탐지
    results = model(frame)
    ball_position = None

    # YOLO 탐지 결과에서 "sports ball" 클래스 확인
    for _, row in results.pandas().xyxy[0].iterrows():
        if row['name'] in ['sports ball', 'ball']:  # 기본 모델 클래스
            x_center = int((row['xmin'] + row['xmax']) / 2)
            y_center = int((row['ymin'] + row['ymax']) / 2)
            ball_position = (x_center, y_center)
            break

    # 공이 탐지되지 않으면 처리 생략
    if ball_position is None:
        return frame, False

    # 스트라이크/볼 판정
    is_strike_result = is_strike(ball_position, strike_zone)

    # 결과를 프레임에 표시
    label = "Strike" if is_strike_result else "Ball"
    color = (0, 255, 0) if is_strike_result else (0, 0, 255)
    # 스트라이크 존 표시
    cv2.rectangle(frame, (strike_zone[0], strike_zone[1]), (strike_zone[2], strike_zone[3]), (255, 255, 255), 2)
    # 공 위치 표시
    cv2.circle(frame, ball_position, 10, color, -1)
    # 결과 텍스트 추가
    cv2.putText(frame, label, (ball_position[0] - 20, ball_position[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame, is_strike_result

def process_video(video_file, output_path, model):
    """
    동영상을 프레임 단위로 처리하고 출력 동영상을 저장합니다.
    """
    # OpenCV로 동영상 로드
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # FPS 기본값 설정
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 동영상 초기화
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        frame, is_strike = process_frame(frame, model)

        # 처리된 프레임 저장
        out.write(frame)

    # 자원 해제
    cap.release()
    out.release()
    os.remove(temp_video_path)
