#===============================================================
# This script detects facial landmarks using a pre-trained model.
#===============================================================

import cv2
import mediapipe as mp
import numpy as np


class Detector:
    def __init__(self):
        """
        Initializes the Mediapipe Detector for pose and face landmark detection.
        """
        self.pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.85
        )        
        self.landmark = None # Stores detected body landmarks
        self.face_landmark = None # Stores detected face landmarks
        self.drawing_landmark = None # Stores landmarks for drawing
        self.width = 0 # Frame width
        self.height = 0 # Frame height

        self.prev_face_sp = None # Previous face starting point
        self.prev_face_ep = None # Previous face ending point

    def process(self, frame):
        """
        Processes an input frame to detect body and face landmarks.
        
        Parameters:
        - frame: Input image frame.
        
        Returns:
        - Boolean indicating if landmarks were detected.
        """
        # Set size
        self.height, self.width = frame.shape[:2]

        # Convert frame to RGB for Mediapipe processing
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        # Detect pose landmarks
        results = self.pose.process(rgb)
        results_face = self.face.process(rgb)
        
        # If no pose or face landmarks detected, return False
        if results.pose_landmarks is None:
            return False
        
        if results_face.multi_face_landmarks is None:
            return False
        
        # Extract and store detected landmarks
        self.landmark = [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]
        self.drawing_landmark = results_face.multi_face_landmarks
        self.face_landmark = [[lm.x * self.width, lm.y * self.height, lm.z] for lm in results_face.multi_face_landmarks[0].landmark]
        return len(self.landmark) > 0


    def get_face_rect(self):
        """
        Computes the bounding box for the detected face.
        
        Returns:
        - Tuple (sp, ep): start point and end point of the bounding box.
        """
        if self.landmark is not None:
            cx, cy = self.landmark[0][0], self.landmark[0][1]
            br = self.landmark[7][0]
            bl = self.landmark[8][0]
            bu = self.landmark[1][1]
            bb = self.landmark[10][1]

            w = (max(br, bl) - min(br, bl)) * 1.5
            h = (max(bu, bb) - min(bu, bb)) * 2.5

            sx = int((cx - w / 2) * self.width)
            sy = int((cy - h / 2) * self.height)
            ex = sx + int(w * self.width)
            ey = sy + int(h * self.height)

            sp, ep = (max(sx, 0), max(sy, 0)), (min(ex, self.width), min(ey, self.height)) # 얼굴 맨 위 좌측 상단, 우측 하단 값(starting point, ending point)

            return sp, ep
        return (0, 0), (1, 1)


    def get_torso_rect(self):
        """
        Computes the bounding box for the torso.
        
        Returns:
        - Tuple (sp, ep): start point and end point of the bounding box.
        """
        if self.landmark is not None:
            sx = int(min(self.landmark[11][0], self.landmark[12][0]) * self.width)
            sy = int(self.height * (self.landmark[9][1] + self.landmark[10][1] + self.landmark[11][1] + self.landmark[12][1]) / 4)
            ex = int(max(self.landmark[11][0], self.landmark[12][0]) * self.width)
            ey = int(min(self.landmark[23][1], self.landmark[24][1]) * self.height)

            return (max(sx, 0), max(sy, 0)), (min(ex, self.width), min(ey, self.height))
        return (0, 0), (1, 1)


    def get_landmark_groups(self):
        """
        Extracts specific landmark groups for eyes, mouth, and eyebrows.
        
        Returns:
        - eye_landmarks: Dictionary of eye landmark points.
        - mouth_landmarks: List of mouth landmark points.
        - eyebrow_landmarks: Dictionary of eyebrow landmark points.
        """
        
        if self.face_landmark is not None:
            # Define indices for eyes, mouth, and eyebrows
            eye_indices = {
                "left_eye": [[33, 133], [160, 144], [159, 145], [158, 153]],
                "right_eye": [[263, 362], [387, 373], [386, 374], [385, 380]]
            }
            
            mouth_indices = [[61, 291], [39, 181], [0, 17], [269, 405]]
            
            eyebrow_indices = {
                "left_eyebrow": [52, 159, 65],
                "right_eyebrow": [295, 386, 282]
            }

            # Fetch actual coordinates for eyes
            eye_landmarks = {
                "left_eye": [[self.face_landmark[i][0:2], self.face_landmark[j][0:2]] for i, j in eye_indices["left_eye"]],
                "right_eye": [[self.face_landmark[i][0:2], self.face_landmark[j][0:2]] for i, j in eye_indices["right_eye"]]
            }

            # Fetch actual coordinates for mouth
            mouth_landmarks = [[self.face_landmark[i][0:2], self.face_landmark[j][0:2]] for i, j in mouth_indices]

            # Fetch actual coordinates for eyebrows
            eyebrow_landmarks = {
                "left_eyebrow": [self.face_landmark[idx][0:2] for idx in eyebrow_indices["left_eyebrow"]],
                "right_eyebrow": [self.face_landmark[idx][0:2] for idx in eyebrow_indices["right_eyebrow"]]
            }

            return eye_landmarks, mouth_landmarks, eyebrow_landmarks
        
        return -1000, -1000, -1000
