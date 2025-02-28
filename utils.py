#===============================================================
# This script contains utility functions for various tasks.
#===============================================================

import numpy as np
import cv2

def draw_signal(signal, width=500, height=150, peaks=None, frame=None, scale=None, ret_scale=False):
    """
    Draws a signal waveform on a given frame (or creates a new one) for visualization.
    """
    # Create empty frame
    if frame is None:
        frame = np.zeros((height, width, 3), np.uint8)
    else:
        height, width = frame.shape[:2]

    # Signal preprocessing
    try:
        np_signal = np.array(signal)
        if scale is not None:
            min_val, max_val = scale
        else:
            max_val, min_val = np_signal.max(), np_signal.min()
        diff_val = max_val - min_val
        np_signal = np_signal if diff_val == 0 else (np_signal - np_signal.min()) / diff_val
    except:
        if ret_scale:
            return frame, (min_val, max_val)
        else:
            return frame

    # Draw signal
    width_offset = width / np_signal.shape[0]
    for i in range(np_signal.shape[0] - 1):
        sx = i * width_offset
        sy = height - (np_signal[i] * height)
        ex = (i + 1) * width_offset
        ey = height - (np_signal[(i + 1)] * height)
        cv2.line(frame, (int(sx), int(sy)), (int(ex), int(ey)), (0, 255, 0), 3)

        if (peaks is not None) and (i in peaks):
            cv2.circle(frame, (int((sx+ex)/2), int((sy+ey)/2)), 5, (0, 0, 255), -1)
    if ret_scale:
        return frame, (min_val, max_val)
    else:
        return frame


def show_signal(name, signal, width=500, height=150, peaks=None):
    """
    Shows the signal in a separate OpenCV window.
    """
    frame = draw_signal(signal, width, height, peaks=peaks)
    cv2.imshow(name, frame)


def draw_custom_landmarks(frame, landmarks_positions, landmark_indices, color=(0, 0, 255)):
    """
    Draws circles on specified landmark indices in the frame.
    """
    for idx in landmark_indices:
        pos = landmarks_positions[idx]
        cv2.circle(frame, (int(pos[0]), int(pos[1])), 1, color, -1)
        

def draw_calibration_text(frame, text1, text2, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.2, color=(0, 0, 255), thickness=3):
    """
    Overlays calibration text on a blurred copy of the input frame.

    Returns:
    - blurred_frame: The blurred frame with the text overlay.
    """

    blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)

    (text1_width, text1_height), _ = cv2.getTextSize(text1, font, font_scale, thickness)
    (text2_width, text2_height), _ = cv2.getTextSize(text2, font, font_scale, thickness)

    center_x = int(frame.shape[1] / 2)
    center_y = int(frame.shape[0] / 2)

    cv2.putText(blurred_frame, text1, 
                (center_x - text1_width // 2, center_y - 10), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(blurred_frame, text2, 
                (center_x - text2_width // 2, center_y + text1_height + 10), 
                font, font_scale, color, thickness, cv2.LINE_AA)
    
    return blurred_frame


def display_features(frame, EBW, EBH, EAR, MAR, lfhf, prediction, fps):
    """
    Displays extracted facial features and other information (drowsiness level, FPS) on the frame.

    Parameters:
    - frame: The image/frame to draw on.
    - EBW: Eyebrow width.
    - EBH: Eyebrow height.
    - EAR: Eye Aspect Ratio.
    - MAR: Mouth Aspect Ratio.
    - lfhf: LF/HF ratio.
    - prediction: Current drowsiness prediction label.
    - fps: Current frames per second.
    """
    cv2.putText(frame, f"EBW: {EBW:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"EBH: {EBH:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame, f"EAR: {EAR:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"MAR: {MAR:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, f"LH/HF: {lfhf:.2f}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Drowsiness Level: {prediction}", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "%02d fps" % round(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


def draw_result(frame, signal, name, bpm, rect, rect_color=(0, 0, 255)):
    """
    Draws the rPPG signal graph and BPM information on the frame.

    Parameters:
    - frame: The image/frame to draw on.
    - signal: The rPPG signal array.
    - name: Title text (e.g. 'rPPG').
    - bpm: Current BPM value.
    - rect: A tuple ((x1,y1),(x2,y2)) bounding the face region.
    - rect_color: Color for the bounding box rectangle.

    Returns:
    - frame: The augmented frame.
    """
    h, w = frame.shape[:2]
    cv2.rectangle(frame, rect[0], rect[1], rect_color, 3)

    bpm_w = int(w / 3)
    signal_frame = draw_signal(signal, width=w-bpm_w)
    bpm_frame = np.zeros((150, bpm_w, 3), np.uint8)

    cv2.putText(bpm_frame, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(bpm_frame, "%03d" % bpm, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)
    frame = np.vstack((frame, np.hstack((signal_frame, bpm_frame))))

    return frame


