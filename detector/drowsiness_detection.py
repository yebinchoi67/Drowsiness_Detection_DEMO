#===============================================================
# This script implements the core drowsiness detection logic.
#===============================================================

from xgboost import XGBClassifier # Import your drowsiness model
import numpy as np
import pandas as pd
import os
import json

class DrowsinessDetector:
    def __init__(self, model_path="model/xgboost_model_weights.json", use_saved_calibration=False, save_calibration=True, calibration_path = '', saved_calibration_path = ''):
        """
        Initializes the DrowsinessDetector class.
        
        Parameters:
        - model_path: Path to the pre-trained Drowsiness detection model.
        - use_saved_calibration: Boolean flag to use previously saved calibration values.
        - save_calibration: Boolean flag to save newly computed calibration values.
        - calibration_path: File path to save calibration values during runtime.
        - saved_calibration_path: File path to load calibration values during runtime.
        """
        self.model = XGBClassifier(random_state=42)
        self.model.load_model(model_path)
        self.calibration_means = {}
        self.calibration_stds = {}
        self.calibration_path = calibration_path
        self.saved_calibration_path = saved_calibration_path
        self.use_saved_calibration = use_saved_calibration
        self.save_calibration = save_calibration

    def calibrate(self, data_buffer):
        """
        Calibrates the feature values using the provided data buffer.
        If previously saved calibration values exist and the flag is enabled, it loads them.
        Otherwise, it computes mean and standard deviation for each feature.
        """
        # Load calibration values if use_saved_calibration is True and the file exists
        if self.use_saved_calibration and os.path.exists(self.saved_calibration_path):
            with open(self.saved_calibration_path, "r") as f:
                calibration_data = json.load(f)
                self.calibration_means = calibration_data["means"]
                self.calibration_stds = calibration_data["stds"]
            print("Loaded calibration values from JSON.")
            return

        # Calculate calibration values
        for feature in ["EAR", "MAR", "EBW", "EBH", "bpm", "lfhf"]:
            feature_values = np.array(data_buffer[feature])

            # Remove zeros from bpm data
            if feature == "bpm":
                feature_values = feature_values[feature_values != 0]

            # Calculate mean and standard deviation
            self.calibration_means[feature] = np.mean(feature_values)
            self.calibration_stds[feature] = np.std(feature_values)

        print("Calibration complete. Means and stds set for each feature.")

        # Save calibration values if save_calibration is True
        if self.save_calibration:
            calibration_data = {
                "means": self.calibration_means,
                "stds": self.calibration_stds
            }
            with open(self.calibration_path, "w") as f:
                json.dump(calibration_data, f)
            print("Calibration values saved to JSON.")
            

    def preprocess(self, data_buffer):
        """
        Preprocesses the input feature values by standardizing them using calibration values.
        
        Parameters:
        - data_buffer: A dictionary containing feature values collected over time.
        
        Returns:
        - feature_df: A Pandas DataFrame containing processed feature values.
        """

        features = {}
        
        # Standardize facial feature values
        for feature in ["EAR", "MAR", "EBW", "EBH"]:
            feature_values = np.array(data_buffer[feature])
            standardized_values = (feature_values - self.calibration_means[feature]) / self.calibration_stds[feature]

            # Compute mean and standard deviation of standardized values
            mean_z = np.mean(standardized_values)
            stddev_z = np.std(standardized_values)

            # Store processed features
            features[f"{feature}_Mean_Z"] = mean_z
            features[f"{feature}_StdDev_meanN"] = stddev_z
            

        # Process PPG (Photoplethysmography) data using centering
        bpm_values = np.array(data_buffer['bpm'])
        bpm_values = bpm_values[bpm_values != 0] # Remove zero values

        meanN_ppg_values = bpm_values - self.calibration_means['bpm']
        mean_bpm = np.mean(meanN_ppg_values)
        stddev_bpm = np.std(meanN_ppg_values)
        
        features["PPG_Mean_meanN"] = mean_bpm
        features["PPG_StdDev_meanN"] = stddev_bpm 
        
        # Store the latest LF/HF ratio
        features["LF_HF_ratio"] = np.array(data_buffer['lfhf'])
        
        # Convert selected features into a Pandas DataFram
        feature_df = pd.DataFrame([{
            'EAR_Mean_Z': features['EAR_Mean_Z'],
            'EBW_Mean_Z': features['EBW_Mean_Z'],
            'EBH_Mean_Z': features['EBH_Mean_Z'],
            'MAR_Mean_Z': features['MAR_Mean_Z'],
            'PPG_StdDev_meanN': features['PPG_StdDev_meanN'],
            'LF_HF_ratio': features['LF_HF_ratio'][-1]
        }])
        
        return feature_df

    def predict(self, data_buffer):
        """
        Predicts the drowsiness state based on the processed feature values.
        
        Parameters:
        - data_buffer: A dictionary containing the most recent feature values.
        
        Returns:
        - prediction_label: A string indicating the predicted state ('Drowsy' or 'Alert').
        """
        processed_data = self.preprocess(data_buffer)
        prediction = self.model.predict(processed_data)
        prediction_label = "Drowsy" if prediction[0] == 1 else "Alert"
        
        return prediction_label

