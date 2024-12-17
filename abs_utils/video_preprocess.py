import cv2

def adjust_brightness_contrast(input_path, output_path, alpha=1.5, beta=50):
    """
    영상의 밝기와 대비를 조절합니다.
    :param input_path: 입력 영상 파일 경로
    :param output_path: 조정된 영상 파일 경로
    :param alpha: 대비 조절값 (1.0 이상으로 설정하면 대비 증가)
    :param beta: 밝기 조절값 (0 이상의 값을 설정하면 밝기 증가)
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 밝기와 대비 조절
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # 저장
        out.write(adjusted_frame)

    cap.release()
    out.release()
    print(f"영상 저장 완료: {output_path}")

def resize_video(input_path, output_path, new_width=1280, new_height=720):
    """
    영상의 해상도를 변경합니다.
    :param input_path: 입력 영상 파일 경로
    :param output_path: 변경된 영상 파일 경로
    :param new_width: 새로운 너비
    :param new_height: 새로운 높이
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 해상도 변경
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # 저장
        out.write(resized_frame)

    cap.release()
    out.release()
    print(f"영상 저장 완료: {output_path}")

def adjust_video(input_path, output_path, alpha=1.3, beta=40, new_width=1920, new_height=1080):
    """
    영상의 조명과 해상도를 동시에 조절합니다.
    :param input_path: 입력 영상 파일 경로
    :param output_path: 조정된 영상 파일 경로
    :param alpha: 대비 조절값
    :param beta: 밝기 조절값
    :param new_width: 새로운 너비
    :param new_height: 새로운 높이
    """
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 밝기와 대비 조절
        adjusted_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # 해상도 변경
        resized_frame = cv2.resize(adjusted_frame, (new_width, new_height))
        
        # 저장
        out.write(resized_frame)

    cap.release()
    out.release()
    print(f"영상 저장 완료: {output_path}")
