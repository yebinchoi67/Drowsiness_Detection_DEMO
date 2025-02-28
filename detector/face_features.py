#===============================================================
# This script extracts facial features related to drowsiness detection.
#===============================================================

import numpy as np

class FaceFeatures:
    MAX_FPS = 30 # Maximum frames per second
    DURATION_RAW = 5  # Number of seconds to keep raw data
    RAW_SIZE = MAX_FPS * DURATION_RAW  # Buffer size for feature history

    def __init__(self):
        """
        Initializes the FaceFeatures class and resets feature buffers.
        """
        self.reset()
 
 
    def reset(self):
        """
        Reset all feature buffers to empty lists.
        """
        self.features = {
            'EAR': [],
            'MAR': [],
            'EBW': [],
            'EBH': []
        }


    def process(self, eye_landmarks, mouth_landmarks, eyebrow_landmarks):
        """
        Process the frame and compute EAR, MAR, EBW, and EBH.
        Stores each feature's value frame-by-frame.
        
        Parameters:
        - eye_landmarks: Dictionary containing eye landmark coordinates.
        - mouth_landmarks: Dictionary containing mouth landmark coordinates.
        - eyebrow_landmarks: Dictionary containing eyebrow landmark coordinates.
        
        Returns:
        - Tuple containing EAR, MAR, EBW, and EBH values for the current frame.
        """
        
        if eye_landmarks == -1000 :
            EAR = -1000
        else :
            EAR = self.calculate_ear(eye_landmarks)
        
        if mouth_landmarks == -1000:
            MAR = -1000
        else :
            MAR = self.calculate_mar(mouth_landmarks)
            
        if eyebrow_landmarks == -1000:
            EBW, EBH = -1000, -1000
        else :
            EBW, EBH = self.calculate_eyebrow_width_height(eyebrow_landmarks)

        # Append raw values to the buffers
        self._update_feature('EAR', EAR)
        self._update_feature('MAR', MAR)
        self._update_feature('EBW', EBW)
        self._update_feature('EBH', EBH)

        # Return the current frame's features for visualization or other processing
        return EAR, MAR, EBW, EBH


    def _update_feature(self, feature_name, value):
        """
        Helper method to update feature buffer with new values per frame.
        Ensures buffer size does not exceed RAW_SIZE.
        """
        self.features[feature_name].append(value)
        if len(self.features[feature_name]) > self.RAW_SIZE:
            self.features[feature_name] = self.features[feature_name][-self.RAW_SIZE:]  # Keep only recent values


    def calculate_ear(self, eye_landmarks):
        """
        Compute the Eye Aspect Ratio (EAR) for both eyes.
        """
        left_ear = self._eye_aspect_ratio(eye_landmarks["left_eye"])
        right_ear = self._eye_aspect_ratio(eye_landmarks["right_eye"])
        return (left_ear + right_ear) / 2


    def calculate_mar(self, mouth_landmarks):
        """
        Compute the Mouth Aspect Ratio (MAR).
        """
        return self._mouth_aspect_ratio(mouth_landmarks)


    def calculate_eyebrow_width_height(self, eyebrow_landmarks):
        """
        Calculate the width between eyebrows (EBW) and average eyebrow height (EBH).
        """
        EBW = self._distance(eyebrow_landmarks["left_eyebrow"][0], eyebrow_landmarks["right_eyebrow"][0])

        # Calculate average height of left and right eyebrows
        left_EBH = (self._distance(eyebrow_landmarks["left_eyebrow"][0], eyebrow_landmarks["left_eyebrow"][1]) +
                    self._distance(eyebrow_landmarks["left_eyebrow"][2], eyebrow_landmarks["left_eyebrow"][1])) / 2
        right_EBH = (self._distance(eyebrow_landmarks["right_eyebrow"][0], eyebrow_landmarks["right_eyebrow"][1]) +
                     self._distance(eyebrow_landmarks["right_eyebrow"][2], eyebrow_landmarks["right_eyebrow"][1])) / 2
        EBH = (left_EBH + right_EBH) / 2  # Average height

        return EBW, EBH


    def _distance(self, p1, p2):
        """
        Compute the Euclidean distance between two points.
        """
        return np.linalg.norm(np.array(p1) - np.array(p2))


    def _eye_aspect_ratio(self, eye):
        """
        Compute the Eye Aspect Ratio (EAR) given eye landmark coordinates.
        """
        N1 = self._distance(eye[1][0], eye[1][1])
        N2 = self._distance(eye[2][0], eye[2][1])
        N3 = self._distance(eye[3][0], eye[3][1])
        D = self._distance(eye[0][0], eye[0][1])
        return (N1 + N2 + N3) / (3 * D)


    def _mouth_aspect_ratio(self, mouth):
        """
        Compute the Mouth Aspect Ratio (MAR) given mouth landmark coordinates.
        """
        N1 = self._distance(mouth[1][0], mouth[1][1])
        N2 = self._distance(mouth[2][0], mouth[2][1])
        N3 = self._distance(mouth[3][0], mouth[3][1])
        D = self._distance(mouth[0][0], mouth[0][1])
        return (N1 + N2 + N3) / (3 * D)


    def get_features(self):
        """
        Retrieve the feature buffers.
        """
        return self.features
