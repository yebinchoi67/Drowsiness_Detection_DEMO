# pyinstaller -D --clean --noconfirm --log-level WARN main.spec

#===============================================================
# This is the main script to run the drowsiness detection system.
#===============================================================

import cv2
import numpy as np
import argparse

from detector.landmark import Detector
from detector.rppg import rPPG
from detector.face_features import FaceFeatures  # Import FaceFeatures
from detector.drowsiness_detection import DrowsinessDetector  # Import your AI model
from timer import Timer
from utils import draw_custom_landmarks, draw_calibration_text, display_features, draw_result
import mediapipe as mp

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def str2bool(v):
    """Custom function to convert strings to boolean values."""
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1', 'True'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0', 'False'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (True/False).")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Drowsiness Detection System")
    parser.add_argument('--use_ppg', type=str2bool, default=True, help='Enable PPG processing')
    parser.add_argument('--use_detect', type=str2bool, default=True, help='Enable landmark detection')
    parser.add_argument('--scale_factor', type=float, default=1.5, help='Window scale factor')
    parser.add_argument('--use_saved_calibration', type=str2bool, default=False, help='Use previously saved calibration data')
    parser.add_argument('--save_calibration', type=str2bool, default=False, help='Save new calibration data')
    parser.add_argument('--calibration_path', type=str, default='calibration/user.json', help='Path to the calibration file')
    parser.add_argument('--saved_calibration_path', type=str, default='calibration/user.json', help='Path to the calibration file')
    return parser.parse_args()


def set_scaled_window(frame_name, frame, scale_factor=1.5):
    """Enable dynamic resizing for the window."""
    cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
    height, width = frame.shape[:2]
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize the window
    cv2.resizeWindow(frame_name, new_width, new_height)
        
        
def main():
    # Parse arguments
    args = parse_arguments()
    
    # Initialize modules
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    # Initialize modules
    detector = Detector() # landmark detector
    rppg = rPPG() 
    face_features = FaceFeatures()
    drowsiness_detector = DrowsinessDetector( 
        use_saved_calibration=args.use_saved_calibration,
        save_calibration=args.save_calibration,
        calibration_path=args.calibration_path,
        saved_calibration_path=args.saved_calibration_path
    ) # Drowsiness model
    
    
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

    first_calibration = True
    prediction = 'Not predicted' # first prediction
    frame_name = 'Drowsiness detection'
    
    
    # Attempt to read the first frame
    ret, frame = cap.read()
    if ret:
        # Set window size to 1.5 times (50% increase) the original resolution
        set_scaled_window(frame_name, frame, scale_factor=args.scale_factor)
    

    while True:
        Timer.set_time_stamp() # Set time
        ret, frame = cap.read() # Get frame
        if not ret:
            break
        visualize_frame = frame.copy()

        # Landmark detection
        if args.use_detect:
            success = detector.process(frame)
            if success:
                face_sp, face_ep = detector.get_face_rect()

                # Draw Mediapipe mesh
                overlay = visualize_frame.copy()
                for face_landmarks in detector.drawing_landmark:
                    mp_drawing.draw_landmarks(
                        image=visualize_frame,
                        landmark_list=face_landmarks,
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            thickness=1, color=(255, 255, 255))
                    )
                alpha = 0.92
                visualize_frame = cv2.addWeighted(overlay, alpha, visualize_frame, 1 - alpha, 0)

                # Draw specific landmarks (eyes, eyebrows, mouth)
                # Eye
                draw_custom_landmarks(visualize_frame, detector.face_landmark,
                                      landmark_indices=[107, 336, 52, 65, 159, 295, 386, 282],
                                      color=(255, 0, 0)) 
                # Mouth
                draw_custom_landmarks(visualize_frame, detector.face_landmark,
                                      landmark_indices=[33, 133, 160, 144, 159, 145, 158, 153,
                                                        263, 362, 387, 373, 386, 374, 385, 380],
                                      color=(255, 255, 0))
                # Eyebrow
                draw_custom_landmarks(visualize_frame, detector.face_landmark,
                                      landmark_indices=[61, 291, 39, 181, 0, 17, 269, 405],
                                      color=(0, 0, 255))

        # PPG processing
        if success:
            if args.use_ppg and ret:
                rppg_signal, raw_ppg, lfhf = rppg.process(frame, face_sp, face_ep)
                rppg_bpm = rppg.get_bpm()

                # Visualize rPPG result
                visualize_frame = draw_result(visualize_frame, rppg_signal, "rPPG",
                                            rppg_bpm, (face_sp, face_ep), (0, 255, 255))
            else:
                raw_ppg = 0
                rppg_bpm = 0
                lfhf = 0
                rppg_signal = []

        # Extract facial features
        eye_landmarks, mouth_landmarks, eyebrow_landmarks = detector.get_landmark_groups()
        EAR, MAR, EBW, EBH = face_features.process(eye_landmarks,
                                                   mouth_landmarks,
                                                   eyebrow_landmarks)

        # Prepare new data for the buffer
        new_data = {
            "EAR": EAR,
            "MAR": MAR,
            "EBW": EBW,
            "EBH": EBH,
            "bpm": rppg_bpm,
            "rawPPG": raw_ppg,
            "lfhf": lfhf
        }
        Timer.buffer_manager(data_buffer, new_data)

        # Display features
        display_features(visualize_frame, EBW, EBH, EAR, MAR, lfhf, prediction, Timer.get_fps())

        # Calibration for the first 30 seconds                     
        if first_calibration:
            if not args.use_saved_calibration:
                visualize_frame = draw_calibration_text(
                    visualize_frame, "Calibration in progress,", "Please wait..."
                )
            if (data_buffer["timestamps"][-1] - data_buffer["timestamps"][0]) >= 29.5:
                drowsiness_detector.calibrate(data_buffer)
                prediction = drowsiness_detector.predict(data_buffer)
                first_calibration = False
        else:
            # Ongoing prediction
            if (data_buffer["timestamps"][-1] - data_buffer["timestamps"][0]) >= 29.5:
                prediction = drowsiness_detector.predict(data_buffer)


        # Display the final result
        cv2.imshow(frame_name, visualize_frame)
        if cv2.getWindowProperty(frame_name, 0) < 0:
            break
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()