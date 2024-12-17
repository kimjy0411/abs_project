import cv2
import torch
import os

# 모델 경로 및 비디오 폴더 설정
MODEL_PATH = "models/best.pt"  # 학습된 YOLO 모델 경로
VIDEO_FOLDER = "videos"        # 비디오 폴더 경로

# 스트라이크 존 판정 함수
def is_strike(ball_position, strike_zone):
    x, y = ball_position
    x_min, y_min, x_max, y_max = strike_zone
    return x_min <= x <= x_max and y_min <= y <= y_max

# 포수와 타자 구분 함수
def detect_catcher_and_batter(objects, image_height):
    """
    탐지된 객체 중 포수를 구분하기 위해 Y 좌표를 기준으로 가장 아래에 있는 객체를 포수로 판단합니다.
    """
    batter_position = None
    catcher_position = None

    for obj in objects:
        label, x_center, y_center = obj['label'], obj['x_center'], obj['y_center']

        # 타자 객체인 경우
        if label == "Batter":
            batter_position = (x_center, y_center)

        # 포수 객체는 타자보다 아래에 위치한다고 가정 (Y 좌표가 크면 화면상 아래)
        if label == "Catcher" or (batter_position and y_center > batter_position[1]):
            catcher_position = (x_center, y_center)

    return batter_position, catcher_position

# 프레임 처리 함수
def process_frame(frame, model):
    results = model(frame)  # YOLO 모델 추론
    objects = []

    # 탐지된 객체 리스트에 추가
    for _, row in results.pandas().xyxy[0].iterrows():
        label = row['name']
        x_center = int((row['xmin'] + row['xmax']) / 2)
        y_center = int((row['ymin'] + row['ymax']) / 2)
        objects.append({'label': label, 'x_center': x_center, 'y_center': y_center})

    # 화면 높이
    height, _, _ = frame.shape

    # 포수와 타자 구분
    batter, catcher = detect_catcher_and_batter(objects, height)

    # 스트라이크 존 설정 (타자의 Y 좌표를 기준으로 설정)
    if batter:
        strike_zone = (
            batter[0] - 50, batter[1] - 100,  # x_min, y_min
            batter[0] + 50, batter[1] + 100   # x_max, y_max
        )
    elif catcher:  # 타자가 없을 경우 포수를 기준으로 설정
        strike_zone = (
            catcher[0] - 50, catcher[1] - 100,
            catcher[0] + 50, catcher[1] + 100
        )
    else:
        return frame  # 타자와 포수를 모두 탐지하지 못한 경우 처리 생략

    # 공 탐지 및 스트라이크/볼 판정
    for obj in objects:
        if obj['label'] == "Baseball_ball":
            ball_position = (obj['x_center'], obj['y_center'])
            is_strike_result = is_strike(ball_position, strike_zone)

            # 결과 표시
            label = "Strike" if is_strike_result else "Ball"
            color = (0, 255, 0) if is_strike_result else (0, 0, 255)
            cv2.rectangle(frame, (strike_zone[0], strike_zone[1]),
                          (strike_zone[2], strike_zone[3]), (255, 255, 255), 2)  # 스트라이크 존
            cv2.circle(frame, ball_position, 10, color, -1)  # 공 위치
            cv2.putText(frame, label, (ball_position[0] - 20, ball_position[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame

# 비디오 처리 함수
def process_video(video_path, output_path, model):
    cap = cv2.VideoCapture(video_path)
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
        frame = process_frame(frame, model)

        # 결과 프레임 저장
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 비디오 처리 완료: {output_path}")

# 메인 함수
def main():
    print("모델 불러오는 중...")
    model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
    print("✅ 모델 불러오기 완료!")

    # 비디오 처리
    video_input = os.path.join(VIDEO_FOLDER, "input_video.mp4")
    video_output = os.path.join(VIDEO_FOLDER, "output_video.mp4")
    print("비디오 처리 중...")
    process_video(video_input, video_output, model)

    print("✅ 스트라이크/볼 판정 완료!")
    print(f"📁 결과 비디오 경로: {video_output}")

if __name__ == "__main__":
    main()
