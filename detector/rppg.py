#===============================================================
# This script processes remote photoplethysmography (rPPG) signals.
#===============================================================

import cv2
import numpy as np
from scipy import signal
from scipy.signal import welch
from timer import Timer


class rPPG:
    MAX_FPS = 30 # Maximum frames per second
    BPM_BAND = (42, 180) # Valid BPM range

    DURATION_RAW = 5 # Duration (seconds) for raw signal storage
    DURATION_VISUALIZE = 5 # Duration (seconds) for visualization buffer
    DURATION_ACCUMULATE_BPM = 1 # Duration (seconds) to accumulate BPM values

    RAW_SIZE = MAX_FPS * DURATION_RAW # Buffer size for raw signals
    VISUALIZE_SIZE = MAX_FPS * DURATION_VISUALIZE # Buffer size for visualization

    def __init__(self):
        """
        Initializes the rPPG class and resets buffers.
        """
        self.reset()


    def reset(self):
        """
        Resets all data buffers.
        """
        self.raw = [] # Stores raw PPG values
        self.visualize = [] # Stores data for visualization
        self.bpm_buffer = [] # Buffer for BPM values
        self.fps_buffer = []  # Buffer to store FPS values
        self.raw_for_lfhf = [] # Stores signal data for LF/HF calculation
        self.bpm = 0 # Current BPM value


    def process(self, frame, sp, ep):
        """
        Processes the given frame to extract rPPG signals and compute BPM.
        
        Parameters:
        - frame: Input video frame.
        - sp: Start point (x, y) for cropping the region of interest.
        - ep: End point (x, y) for cropping the region of interest.
        
        Returns:
        - Visualization signal buffer
        - Most recent PPG signal value
        - LF/HF ratio (if available, otherwise 0)
        """
        # Calculate signal value
        crop = frame[sp[1]: ep[1], sp[0]: ep[0], ...] # Extract face region
        mask = self._get_skin_mask(crop) # Get skin mask
        val = self._calculate_ppg_value(crop, mask) # Compute PPG signal value

        # Append buffer
        self.raw.append(val)
        self.raw = self.raw[-self.RAW_SIZE:]

        # Processing
        if len(self.raw) == self.RAW_SIZE:
            fps = Timer.get_fps()
            self.fps_buffer.append(fps)
            if len(self.fps_buffer) < 30:
                pass
            else:
                self.fps_buffer = self.fps_buffer[-30:] # Keep latest 30 FPS values
            
            # Refine signal
            raw_signal = np.array(self.raw[-int(self.DURATION_RAW * fps):]).transpose()
            detrended = self._detrend_signal(raw_signal, fps) # Remove trends
            bandpassed = self._filter_bandpass(-detrended, fps, self.BPM_BAND) # Apply bandpass filter
            
            # Accumulate signal data for LF/HF ratio calculation
            self.raw_for_lfhf.extend(bandpassed[-int(fps):])
            self.raw_for_lfhf = self.raw_for_lfhf[-int(sum(self.fps_buffer)):]

            # Calculate bpm
            bpm = self._get_bpm(bandpassed, fps)
            self.bpm_buffer.append(bpm)
            self.bpm_buffer = self.bpm_buffer[-int(self.DURATION_ACCUMULATE_BPM * fps):]

            # Visualization
            self.visualize.append(bandpassed[-1])
            self.visualize = self.visualize[-self.VISUALIZE_SIZE:]


            # Compute LF/HF ratio if enough data has accumulated
            if len(self.fps_buffer) >= 30:  
                avg_fps = np.mean(self.fps_buffer)
                lf_hf_ratio = self._calculate_lf_hf_ratio(self.raw_for_lfhf, avg_fps)  
                return self.visualize, bandpassed[-1], lf_hf_ratio
            
            return self.visualize, bandpassed[-1], 0
        else:
            return [0] * self.VISUALIZE_SIZE, 0, 0

    
    def _get_skin_mask(self, image, n_constant=0):
        """
        Generates a binary skin mask from the input image using YCrCb color space.
        """
        try:
            low = np.array([0, 133, 77], np.uint8)
            high = np.array([235, 173, 127], np.uint8)

            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            # 특정 부분으로 threshold값 주면 피부로 인식(피부값만 인식해서 마스킹)
            mask = cv2.inRange(ycrcb, low, high)
            mask[mask == 255] = 1

            return mask
        except Exception as e:
            return None

    def _calculate_ppg_value(self, image, mask): 
        """
        Computes the PPG value by averaging Cr and Cb channels over the masked region.
        """
        try:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            _, cr, cb = cv2.split(ycrcb)

            if not isinstance(mask, type(None)):
                n_pixels = image.shape[0] * image.shape[1]
            else:
                n_pixels = max(1, np.sum(mask))
                cr[mask == 0] = 0
                cb[mask == 0] = 0
            return (np.sum(cr) + np.sum(cb)) / n_pixels
        except Exception as e:
            return self.raw[-1] if len(self.raw) > 0 else 0.0


    def _detrend_signal(self, arr, wsize):
        """
        Removes trends from the signal using a moving average filter.
        """
        try:
            if not isinstance(wsize, int):
                wsize = int(wsize)
            norm = np.convolve(np.ones(len(arr)), np.ones(wsize), mode='same')
            mean = np.convolve(arr, np.ones(wsize), mode='same') / norm
            return (arr - mean) / (mean + 1e-15)
        except ValueError:
            return arr

    def _filter_bandpass(self, arr, srate, band): 
        """
        Applies a bandpass filter to isolate valid heart rate frequencies.
        """
        try:
            nyq = 60 * srate / 2
            coef_vector = signal.butter(5, [band[0] / nyq, band[1] / nyq], 'bandpass')
            return signal.filtfilt(*coef_vector, arr)
        except ValueError:
            return arr

    def _get_bpm(self, arr, fps):
        """
        Computes the heart rate (BPM) using FFT analysis.
        """
        try:
            windowed_arr = arr * np.hanning(len(arr))
            signal_len = len(windowed_arr)

            pad_factor = max(1.0, (60 * fps) / signal_len)
            n_padded = int(len(windowed_arr) * pad_factor)

            fft = np.fft.rfft(windowed_arr, n=n_padded)
            f = np.fft.rfftfreq(n_padded, d=1 / fps)

            frequency_spectrum = np.abs(fft)

            fundamental_peak = np.argmax(frequency_spectrum)

            bpm = int(f[fundamental_peak] * 60)
            bpm = np.clip(bpm, self.BPM_BAND[0], self.BPM_BAND[1]).item()
            return bpm
        except (ValueError, FloatingPointError):
            return 0


    def get_bpm(self):
        """
        Computes the final BPM value using a trimmed mean approach.
        
        Returns:
        - The computed BPM value.
        """
        if Timer.check_sec_ppg():
            if len(self.bpm_buffer) > 0:
                sorted_bpm = np.sort(self.bpm_buffer)
                bpm_len = len(sorted_bpm) // 3
                self.bpm = round(sorted_bpm[bpm_len: -bpm_len].mean())

                return self.bpm
        return self.bpm
    
    
    def _calculate_lf_hf_ratio(self, signal, sampling_rate):
        """
        Computes the LF/HF ratio using Welch's method.
        """
        f, psd = welch(signal, fs=sampling_rate)
        LF_band = (f >= 0.04) & (f < 0.15)
        HF_band = (f >= 0.15) & (f < 0.4)

        LF = np.trapz(psd[LF_band], f[LF_band])
        HF = np.trapz(psd[HF_band], f[HF_band])
        
        # Calculate LF/HF ratio
        LF_HF_ratio = LF / HF if HF != 0 else np.nan
        return LF_HF_ratio