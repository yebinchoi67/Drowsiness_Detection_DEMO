#===============================================================
# This script provides time-related utility functions.
#===============================================================

from time import perf_counter

class Timer:
    """
    Utility class for managing time-based calculations such as FPS tracking,
    periodic checks, and buffer management.
    """
    time_stamps = [] # Stores timestamps of recent frames
    window_size = 100 # The number of timestamps to keep for FPS calculation
    fps = 30 # Default frames per second

    rppg_timer_t = 0 # Timer for rPPG processing interval
    rrsp_timer_t = 0 # Timer for respiration processing interval
    lf_hf_timer_t = 0  # Timer for LF/HF ratio computation interval

    @classmethod
    def set_time_stamp(cls): 
        """
        Records the current timestamp and updates the estimated FPS.
        The FPS is calculated based on the timestamps of the most recent frames.
        """
        cls.time_stamps.append(perf_counter())
        cls.time_stamps = cls.time_stamps[-cls.window_size:]
        cls.fps = 30 if len(cls.time_stamps) == 1 else (len(cls.time_stamps) - 1) / (cls.time_stamps[-1] - cls.time_stamps[0])

    @classmethod
    def get_fps(cls):
        """
        Returns the current estimated FPS based on recent timestamps.
        """
        return cls.fps


    #@classmethod
    @staticmethod
    def buffer_manager(data_buffer, new_data):
        """
        Manages a data buffer by appending new values and maintaining a 30-second window.
        
        Parameters:
        - data_buffer: Dictionary storing accumulated data with timestamps.
        - new_data: Dictionary containing new values to be appended.
        """
        current_time = perf_counter()  # Get current time

        # Append new data to the buffer
        for key in new_data:
            data_buffer[key].append(new_data[key])
        data_buffer["timestamps"].append(current_time)

        # Determine the number of old entries to remove (older than 30 seconds)
        remove_count = 0
        for timestamp in data_buffer["timestamps"]:
            if current_time - timestamp > 30:
                remove_count += 1
            else:
                break
        
        # Remove outdated entries from each key in the buffer
        if remove_count > 0:
            for key in data_buffer:
                data_buffer[key] = data_buffer[key][remove_count:]


    @classmethod
    def check_sec_ppg(cls):
        curr_t = perf_counter()

        if cls.rppg_timer_t == 0:
            cls.rppg_timer_t = curr_t
            return True
        elif (curr_t - cls.rppg_timer_t) > 1:
            cls.rppg_timer_t = curr_t
            return True
        else:
            return False

    @classmethod
    def check_sec_rsp(cls):
        curr_t = perf_counter()

        if cls.rrsp_timer_t == 0:
            cls.rrsp_timer_t = curr_t
            return True
        elif (curr_t - cls.rrsp_timer_t) > 1:
            cls.rrsp_timer_t = curr_t
            return True
        else:
            return False

