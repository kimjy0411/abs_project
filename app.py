import cv2
import os
import torch

# 모델 불러오기
MODEL_PATH = 'models/best2.pt'  # YOLO 모델 파일 경로
VIDEOS_FOLDER = 'videos'  # 비디오 폴더 경로

print("모델 불러오는 중...")
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
print("✅ 모델 불러오기 완료!")

# 스트라이크 존 설정 (고정된 값: 이미지 좌표 기준)
STRIKE_ZONE = (1000, 512, 1192, 674)  # (x_min, y_min, x_max, y_max)

def is_strike(ball_position, strike_zone):
    """공의 위치가 스트라이크 존에 있는지 확인합니다."""
    x, y = ball_position
    x_min, y_min, x_max, y_max = strike_zone
    return x_min <= x <= x_max and y_min <= y <= y_max

def process_frame(frame, ball_inside, last_position, last_label):
    """프레임을 처리하고 스트라이크/볼 판정을 수행합니다."""
    ball_position = None

    # YOLO 모델로 객체 탐지
    results = model(frame)

    for _, row in results.pandas().xyxy[0].iterrows():
        class_name = row['name']
        if class_name == 'Baseball_ball':  # 공 탐지
            x_center = int((row['xmin'] + row['xmax']) / 2)
            y_center = int((row['ymin'] + row['ymax']) / 2)
            ball_position = (x_center, y_center)
            last_position = ball_position  # 마지막 위치 업데이트
            break

    # 공이 탐지되었는지 확인
    if ball_position:
        is_strike_result = is_strike(ball_position, STRIKE_ZONE)
        label = "Ball"  # 기본값
        color = (0, 0, 255)  # 빨간색

        if is_strike_result:
            label = "Strike"
            color = (0, 255, 0)  # 초록색

        ball_inside = True  # 공이 탐지된 상태

        # 시각화
        cv2.rectangle(frame, (STRIKE_ZONE[0], STRIKE_ZONE[1]), (STRIKE_ZONE[2], STRIKE_ZONE[3]), (255, 255, 255), 2)
        cv2.circle(frame, ball_position, 10, color, -1)
        cv2.putText(frame, label, (ball_position[0] - 20, ball_position[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        last_label = None  # 탐지 중이므로 판정 초기화
    else:
        # 공이 탐지되지 않을 때 마지막 위치를 기준으로 판정
        if last_position:
            if is_strike(last_position, STRIKE_ZONE):
                last_label = "STRIKE"
            else:
                last_label = "BALL"
            last_position = None  # 판정 후 마지막 위치 초기화

    # 화면 중앙에 판정 결과 출력
    if last_label:
        color = (0, 255, 0) if last_label == "STRIKE" else (0, 0, 255)
        cv2.putText(frame, last_label, (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)

    return frame, ball_inside, last_position, last_label

def process_video(video_path):
    """비디오 파일을 처리하고 실시간으로 출력합니다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⚠️ 비디오를 열 수 없습니다.")
        return

    resize_width, resize_height = 640, 480  # 화면 크기 조정
    ball_inside = False  # 공이 스트라이크 존 안에 있는지 상태 변수
    last_position = None  # 공의 마지막 위치
    last_label = None  # 최종 판정 결과

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        processed_frame, ball_inside, last_position, last_label = process_frame(
            frame, ball_inside, last_position, last_label
        )

        # 화면 크기 조정
        resized_frame = cv2.resize(processed_frame, (resize_width, resize_height))

        # 화면에 표시
        cv2.imshow("Strike/Ball Detection", resized_frame)

        # 종료 조건: ESC 키 누르면 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    print("✅ 실시간 스트라이크/볼 판정을 시작합니다.")
    print("📂 비디오 폴더에서 영상을 불러옵니다...")

    # 비디오 폴더 내 파일 목록 가져오기
    video_files = [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print("⚠️ 비디오 폴더에 영상 파일이 없습니다.")
        return

    # 비디오 선택 및 처리
    for i, video_file in enumerate(video_files):
        print(f"{i + 1}. {video_file}")
    choice = int(input("재생할 비디오 번호를 입력하세요: ")) - 1

    if 0 <= choice < len(video_files):
        selected_video = os.path.join(VIDEOS_FOLDER, video_files[choice])
        process_video(selected_video)
    else:
        print("⚠️ 잘못된 입력입니다.")

if __name__ == "__main__":
    main()
