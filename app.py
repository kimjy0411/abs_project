import streamlit as st
from abs_utils.video_utils import process_video
from abs_utils.video_preprocess import adjust_video
from abs_utils.model_utils import load_yolo_model
import tempfile
import os

# Streamlit UI
st.title("실시간 스트라이크/볼 판정 시스템")

# 모델 로드
st.info("모델 로드 중...")
model = load_yolo_model()  # 모델 로드 함수
st.success("모델 로드 완료!")

# 동영상 업로드
uploaded_video = st.file_uploader("야구 경기 동영상을 업로드하세요", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # 동영상을 임시 파일로 저장
    temp_input_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input_file.write(uploaded_video.read())

    st.video(temp_input_file.name)

    # 영상 조정 설정
    st.sidebar.header("영상 조정 설정")
    alpha = st.sidebar.slider("대비 (alpha)", 1.0, 3.0, 1.5, 0.1)
    beta = st.sidebar.slider("밝기 (beta)", 0, 100, 50, 5)
    width = st.sidebar.number_input("너비 (px)", 640, 1920, 1280, 10)
    height = st.sidebar.number_input("높이 (px)", 480, 1080, 720, 10)

    # 영상 조정 버튼
    adjusted_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    if st.sidebar.button("영상 조정 적용"):
        with st.spinner("영상 조정 중..."):
            adjust_video(temp_input_file.name, adjusted_output.name, alpha=alpha, beta=beta, new_width=width, new_height=height)
        st.success("영상 조정이 완료되었습니다!")
        st.video(adjusted_output.name)

    # 스트라이크/볼 판정 시작
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    if st.button("스트라이크/볼 판정 시작"):
        with st.spinner("동영상 처리 중..."):
            process_video(adjusted_output.name, temp_output_file.name, model)
        st.success("스트라이크/볼 판정이 완료되었습니다!")
        st.video(temp_output_file.name)

    # 임시 파일 삭제
    os.remove(temp_input_file.name)
    os.remove(adjusted_output.name)
    os.remove(temp_output_file.name)
