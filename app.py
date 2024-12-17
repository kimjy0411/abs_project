import cv2
import os
import torch
import time

# 모델 불러오기
MODEL_PATH = 'models/best2.pt'  # YOLO 모델 파일 경로
VIDEOS_FOLDER = 'videos'  # 비디오 폴더 경로

print("모델 불러오는 중...")
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
print("✅ 모델 불러오기 완료!")

# 스트라이크 존 설정 (고정된 값: 이미지 좌표 기준)
STRIKE_ZONE = (1000, 472, 1192, 634)  # (x_min, y_min - 20, x_max, y_max - 20)
  

def is_strike(ball_position, strike_zone):
    """공의 위치가 스트라이크 존에 있는지 확인합니다."""
    x, y = ball_position
    x_min, y_min, x_max, y_max = strike_zone
    return x_min <= x <= x_max and y_min <= y <= y_max

def process_frame(frame, ball_inside, last_position, last_label, display_timer, detection_timer):
    """프레임을 처리하고 스트라이크/볼 판정을 수행합니다."""
    ball_position = None
    results = model(frame)

    # YOLO 모델로 객체 탐지
    for _, row in results.pandas().xyxy[0].iterrows():
        class_name = row['name']
        if class_name == 'Baseball_ball':  # 공 탐지
            x_center = int((row['xmin'] + row['xmax']) / 2)
            y_center = int((row['ymin'] + row['ymax']) / 2)
            ball_position = (x_center, y_center)
            last_position = ball_position  # 마지막 위치 업데이트
            ball_inside = True  # 탐지 중 상태 설정
            detection_timer = time.time()  # 탐지 시점 타이머 업데이트
            break

    # 공이 탐지 중일 때
    if ball_position:
        display_timer = None  # 문구 표시 타이머 초기화
        last_label = None     # 탐지 중이므로 문구 초기화

        # 스트라이크 존 시각화
        cv2.rectangle(frame, (STRIKE_ZONE[0], STRIKE_ZONE[1]), (STRIKE_ZONE[2], STRIKE_ZONE[3]), (255, 255, 255), 2)
        cv2.circle(frame, ball_position, 10, (0, 255, 0), -1)  # 초록색 원 (탐지 중)
    else:
        # 탐지가 끊어진 경우: 0.5초 동안 탐지되지 않으면 종료 처리
        if ball_inside and time.time() - detection_timer > 0.5:
            ball_inside = False  # 탐지 종료 상태 설정
            if last_position:  # 마지막 좌표로 판정 수행
                if is_strike(last_position, STRIKE_ZONE):
                    last_label = "STRIKE"
                else:
                    last_label = "BALL"
                display_timer = time.time()  # 문구 표시 타이머 설정
            last_position = None  # 마지막 좌표 초기화

    # 문구를 1초 동안 표시
    if last_label and display_timer:
        if time.time() - display_timer < 1:  # 1초 동안 표시
            color = (0, 255, 0) if last_label == "STRIKE" else (0, 0, 255)
            cv2.putText(frame, last_label, (300, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5)
        else:
            last_label = None  # 1초 후 문구 제거

    return frame, ball_inside, last_position, last_label, display_timer, detection_timer

def process_video(video_path):
    """비디오 파일을 처리하고 실시간으로 출력합니다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⚠️ 비디오를 열 수 없습니다.")
        return

    resize_width, resize_height = 640, 480  # 화면 크기 조정
    ball_inside = False  # 공이 탐지된 상태인지 확인
    last_position = None  # 공의 마지막 위치
    last_label = None  # 최종 판정 결과
    display_timer = None  # 문구 표시 타이머
    detection_timer = time.time()  # 탐지 상태 타이머

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 프레임 처리
        processed_frame, ball_inside, last_position, last_label, display_timer, detection_timer = process_frame(
            frame, ball_inside, last_position, last_label, display_timer, detection_timer
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
