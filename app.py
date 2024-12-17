import cv2
import torch
import os

# 모델 경로 및 비디오 폴더 설정
MODEL_PATH = "models/best_backup.pt"
VIDEO_FOLDER = "videos"

# YOLO 모델 불러오기
print("모델 불러오는 중...")
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH, force_reload=True)
print("✅ 모델 불러오기 완료!")


def calculate_strike_zone(batter_box):
    """
    타자의 객체를 기반으로 스트라이크 존을 계산합니다.
    :param batter_box: 타자 바운딩 박스 (xmin, ymin, xmax, ymax)
    :return: 스트라이크 존 좌표 (x_min, y_min, x_max, y_max)
    """
    xmin, ymin, xmax, ymax = batter_box
    height = ymax - ymin
    # 스트라이크 존은 타자의 상체 부분을 기준으로 함 (대략 30% ~ 70% 높이)
    strike_zone_y_min = int(ymin + 0.3 * height)
    strike_zone_y_max = int(ymin + 0.7 * height)
    strike_zone_x_min = int(xmin)
    strike_zone_x_max = int(xmax)

    return (strike_zone_x_min, strike_zone_y_min, strike_zone_x_max, strike_zone_y_max)


def is_strike(ball_box, strike_zone):
    """
    공의 바운딩 박스가 스트라이크 존에 포함되는지 판정합니다.
    :param ball_box: 공 바운딩 박스 (xmin, ymin, xmax, ymax)
    :param strike_zone: 스트라이크 존 좌표
    :return: True (스트라이크), False (볼)
    """
    ball_xmin, ball_ymin, ball_xmax, ball_ymax = ball_box
    sx_min, sy_min, sx_max, sy_max = strike_zone

    # 공의 중심 좌표
    ball_center_x = (ball_xmin + ball_xmax) // 2
    ball_center_y = (ball_ymin + ball_ymax) // 2

    # 스트라이크 존 안에 공의 중심이 있는지 확인
    return sx_min <= ball_center_x <= sx_max and sy_min <= ball_center_y <= sy_max


def process_video(video_path):
    """
    비디오 파일을 읽어 실시간으로 스트라이크/볼 판정을 수행합니다.
    :param video_path: 비디오 파일 경로
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 비디오 파일을 열 수 없습니다: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 모델로 객체 탐지 수행
        results = model(frame)

        # 탐지된 객체에서 타자와 공 확인
        batter_box = None
        ball_box = None

        for *box, conf, cls_id in results.xyxy[0]:
            label = model.names[int(cls_id)]
            if label == "Batter":
                batter_box = list(map(int, box))
            elif label == "Baseball_ball":
                ball_box = list(map(int, box))

        # 스트라이크 존 계산 및 표시
        if batter_box:
            strike_zone = calculate_strike_zone(batter_box)
            cv2.rectangle(frame, (strike_zone[0], strike_zone[1]), (strike_zone[2], strike_zone[3]), (255, 255, 255), 2)
            cv2.putText(frame, "Strike Zone", (strike_zone[0], strike_zone[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # 공이 탐지되면 스트라이크/볼 판정
            if ball_box:
                is_strike_result = is_strike(ball_box, strike_zone)
                label = "Strike" if is_strike_result else "Ball"
                color = (0, 255, 0) if is_strike_result else (0, 0, 255)

                # 공 표시
                cv2.rectangle(frame, (ball_box[0], ball_box[1]), (ball_box[2], ball_box[3]), color, 2)
                cv2.putText(frame, label, (ball_box[0], ball_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # 화면 출력
        cv2.imshow("Strike/Ball Detection", frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    # 비디오 폴더의 모든 파일 확인
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.avi', '.mov'))]

    if not video_files:
        print("❌ videos 폴더에 비디오 파일이 없습니다.")
        return

    print("🎥 사용할 비디오 목록:")
    for idx, file in enumerate(video_files):
        print(f"[{idx}] {file}")

    # 비디오 선택
    choice = int(input("처리할 비디오 번호를 입력하세요: "))
    if 0 <= choice < len(video_files):
        video_path = os.path.join(VIDEO_FOLDER, video_files[choice])
        print(f"▶ '{video_files[choice]}' 처리를 시작합니다...")
        process_video(video_path)
    else:
        print("❌ 올바른 비디오 번호를 선택해주세요.")


if __name__ == "__main__":
    main()
