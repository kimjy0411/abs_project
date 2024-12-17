import streamlit as st
from abs_utils.video_utils import process_video
from abs_utils.model_utils import load_yolo_model
import tempfile
import os

# Streamlit UI
st.title("실시간 스트라이크/볼 판정 시스템")

# 모델 로드
st.info("모델 로드 중...")
model = load_yolo_model()
st.success("모델 로드 완료!")

# 동영상 업로드
uploaded_video = st.file_uploader("야구 경기 동영상을 업로드하세요", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # 동영상을 임시 파일로 저장
    temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input_file.write(uploaded_video.read())

    st.video(temp_input_file.name)

    # 출력 파일 설정
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_path = temp_output_file.name

    # 동영상 처리 및 판정 시작
    if st.button("스트라이크/볼 판정 시작"):
        with st.spinner("동영상 처리 중..."):
            process_video(temp_input_file.name, output_path, model)
        st.success("스트라이크/볼 판정이 완료되었습니다!")
        st.video(output_path)

    # 임시 파일 삭제
    os.remove(temp_input_file.name)
    os.remove(temp_output_file.name)
