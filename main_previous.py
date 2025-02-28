# pyinstaller -D --clean --noconfirm --log-level WARN main.spec

import cv2
import numpy as np

from detector.landmark import Detector
from detector.rppg import rPPG
from detector.face_features import FaceFeatures  # Import FaceFeatures
from detector.drowsiness_detection import DrowsinessDetector  # Import your AI model
from timer import Timer
from utils import draw_signal
import mediapipe as mp
# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles



def draw_result(frame, signal, name, bpm, rect, rect_color=(0, 0, 255)):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, rect[0], rect[1], rect_color, 3)

    bpm_w = int(w / 3)

    signal_frame = draw_signal(signal, width=w-bpm_w)
    bpm_frame = np.zeros((150, bpm_w, 3), np.uint8)

    cv2.putText(bpm_frame, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bpm_frame, "%03d" % bpm, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
    frame = np.vstack((frame, np.hstack((signal_frame, bpm_frame))))

    return frame

def draw_custom_landmarks(frame, landmarks_positions, landmark_indices, color=(0, 0, 255)):
    for idx in landmark_indices:
        pos = landmarks_positions[idx]
        cv2.circle(frame, (int(pos[0]), int(pos[1])), 1, color, -1)
        

def draw_calibration_text(frame, text1, text2, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.2, color=(0, 0, 255), thickness=3):
    # 화면을 흐리게 만들기
    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

    # 텍스트 크기 측정
    (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_scale, thickness)
    (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_scale, thickness)

    # 프레임 중앙 좌표 계산
    center_x = int(frame.shape[1] / 2)
    center_y = int(frame.shape[0] / 2)

    # 텍스트를 중앙에 배치하여 표시
    cv2.putText(blurred_frame, text1, 
                (center_x - text1_width // 2, center_y - 10), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(blurred_frame, text2, 
                (center_x - text2_width // 2, center_y + text1_height + 10), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    
    return blurred_frame

def display_features(frame, EBW, EBH, EAR, MAR, lfhf, prediction, fps):
    """Display extracted features and other information on the frame."""
    cv2.putText(frame, f"EBW: {EBW:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"EBH: {EBH:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"EAR: {EAR:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"MAR: {MAR:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"LH/HF: {lfhf:.2f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Drowsiness Level: {prediction}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "%02d fps" % round(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

def set_scaled_window(frame_name, frame, scale_factor=1.5):
    # Enable dynamic resizing for the window
    cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
    
    # Get the resolution of the frame
    height, width = frame.shape[:2]
    
    # Scale the width and height by the specified factor
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize the window
    cv2.resizeWindow(frame_name, new_width, new_height)
        
def main():
    # Initialize modules
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    detector = Detector() # 랜드마크 detector
    rppg = rPPG() # PPG
    face_features = FaceFeatures()  # 얼굴 랜드마크 피처
    drowsiness_detector = DrowsinessDetector(use_saved_calibration=False, save_calibration=False)  # Drowsiness model

    # initiate prediction
    prediction = 'Not predicted'
    
    # Data buffer to store 30 seconds of data
    data_buffer = {
        "timestamps": [],
        "EAR": [],
        "MAR": [],
        "EBW": [],
        "EBH": [],
        "bpm": [],
        "rawPPG": [],
        "lfhf" : []
    }

    # Set flags
    use_ppg = True
    use_detect = True
    
    # 캘리브레이션 용 
    first_calibration = True

    
    # Set frame
    frame_name = 'Drowsiness detection'
    #cv2.namedWindow(frame_name)
    
    # Capture the first frame to get the original resolution
    ret, frame = cap.read()
    if ret:
        # Set window size to 1.5 times (50% increase) the original resolution
        set_scaled_window(frame_name, frame, scale_factor=1.5)
    
    #########################시작##############################
    while True:
        # Set time
        Timer.set_time_stamp()

        # Get frame
        ret, frame = cap.read()
        if not ret:
            break
        visualize_frame = frame.copy()
        
        #################################################3
        # Calculate landmark
        if use_detect:
            ret = detector.process(frame)

            ## 랜드마크 그리기
            overlay = visualize_frame.copy()
            connection_spec = mp_drawing.DrawingSpec(thickness=1, color=(255, 255, 255))  # 투명한 하얀색 선

            # Main drawing function
            for face_landmarks in detector.drawing_landmark:
                mp_drawing.draw_landmarks(
                    image=visualize_frame,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=connection_spec)        
                
            # Blend the overlay with the original frame using transparency
            alpha = 0.92  # Set transparency level (0.0 to 1.0)
            visualize_frame = cv2.addWeighted(overlay, alpha, visualize_frame, 1 - alpha, 0)

            # 눈썹 랜드마크 그리기
            draw_custom_landmarks(visualize_frame, detector.face_landmark, landmark_indices=[107, 336, 52, 65, 159, 295, 386, 282], color=(255, 0, 0))
            
            # 눈 랜드마크 그리기
            draw_custom_landmarks(visualize_frame, detector.face_landmark, landmark_indices=[33, 133, 160, 144, 159, 145, 158, 153, 263, 362, 387, 373, 386, 374, 385, 380], color=(255, 255, 0))
            
            # 입 랜드마크 그리기
            draw_custom_landmarks(visualize_frame, detector.face_landmark, landmark_indices=[61, 291, 39, 181, 0, 17, 269, 405], color=(0, 0, 255))
            
        ###################################################  
        # 피처 계산 및 예측  
        if ret:
            if use_ppg:
                # Get landmark
                face_sp, face_ep = detector.get_face_rect()

                # PPG processing
                rppg_signal, raw_ppg, lfhf = rppg.process(frame, face_sp, face_ep)
                rppg_bpm = rppg.get_bpm()

                # Visualize rPG
                visualize_frame = draw_result(visualize_frame, rppg_signal, "rPPG", rppg_bpm, (face_sp, face_ep), (0, 255, 255))

            # 랜드마크 기반 얼굴 피처 계산
            eye_landmarks, mouth_landmarks, eyebrow_landmarks = detector.get_landmark_groups()
            # Calculate EAR, MAR, EBW, EBH for each frame
            EAR, MAR, EBW, EBH = face_features.process(eye_landmarks, mouth_landmarks, eyebrow_landmarks)    
                        
            # 현재 프레임의 특징값을 새로운 데이터로 정리
            new_data = {
                "EAR": EAR,
                "MAR": MAR,
                "EBW": EBW,
                "EBH": EBH,
                "bpm": rppg_bpm,
                "rawPPG": raw_ppg,
                "lfhf": lfhf
            }

            # buffer_manager로 데이터 버퍼를 관리하여 30초 데이터만 유지
            Timer.buffer_manager(data_buffer, new_data)
            
            # Call the display function with relevant parameters
            display_features(
                visualize_frame,
                EBW=EBW,
                EBH=EBH,
                EAR=EAR,
                MAR=MAR,
                lfhf=lfhf,
                prediction=prediction,
                fps=Timer.get_fps())
            
            
            if first_calibration:
                # calibration 중 화면 표시
                visualize_frame = draw_calibration_text(visualize_frame, "Calibration in progress,", "Please wait...")
                
                ## 캘리브레이션용 & 졸음 예측 시작
                if (data_buffer["timestamps"][-1] - data_buffer["timestamps"][0]) >= 29.5: #30초 데이터 모이면 들어감
                    drowsiness_detector.calibrate(data_buffer)
                    prediction = drowsiness_detector.predict(data_buffer)
        
                    first_calibration = False
            
            ## 졸음 예측        
            if (data_buffer["timestamps"][-1] - data_buffer["timestamps"][0]) >= 29.5: #30초 데이터 모이면 들어감
                    prediction = drowsiness_detector.predict(data_buffer)
            

        # Close event
        try:
            if cv2.getWindowProperty(frame_name, 0) < 0:
                break
        except:
            break

        cv2.imshow(frame_name, visualize_frame)
        key = cv2.waitKey(1)

        if key == 27:
            break
        elif key == ord(' '):
            use_detect = not use_detect

    cv2.destroyAllWindows()
    cap.release()


if __name__ == '__main__':
    main()

