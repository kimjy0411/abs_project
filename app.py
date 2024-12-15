import streamlit as st
from abs_utils.video_utils import process_frame
from abs_utils.model_utils import load_yolo_model
import cv2
import tempfile
import time

# Streamlit UI
st.title("실시간 스트라이크/볼 판정 시스템")

# 모델 로드
st.info("모델 로드 중...")
model = load_yolo_model()  # 모델 로드
st.success("모델 로드 완료!")

# 동영상 업로드
uploaded_video = st.file_uploader("야구 경기 동영상을 업로드하세요", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # 동영상을 임시 파일로 저장
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_video.read())

    st.video(temp_file.name)

    # 동영상 처리 및 판정 시작
    if st.button("스트라이크/볼 판정 시작"):
        cap = cv2.VideoCapture(temp_file.name)
        frame_placeholder = st.empty()
        status_message = st.empty()

        # 카운트 초기화
        strike_count = 0
        ball_count = 0
        total_pitches = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 프레임 처리 (스트라이크/볼 판정)
            frame, is_strike = process_frame(frame, model)

            # 투구 결과 업데이트
            total_pitches += 1
            if is_strike:
                strike_count += 1
            else:
                ball_count += 1

            # 상태 업데이트
            status_message.markdown(f"""
            **투구 수**: {total_pitches}  
            **스트라이크**: {strike_count}  
            **볼**: {ball_count}
            """)

            # 프레임 표시
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_bgr, channels="RGB")

            # 처리 속도 조절 (프레임 속도에 맞추기)
            time.sleep(1 / cap.get(cv2.CAP_PROP_FPS))

        cap.release()
        st.success("스트라이크/볼 판정이 완료되었습니다!")
