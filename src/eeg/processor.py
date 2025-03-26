"""
EEG data collection and processing functionality.
"""

import numpy as np
import time
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError, BrainFlowPresets
from brainflow.data_filter import DataFilter, FilterTypes
import threading
import queue

from src.utils.config import (
    FREQ_BANDS, WINDOW_SIZE, LINE_NOISE_FREQ, 
    SIMULATED_SAMPLING_RATE, SIGNAL_THRESHOLD
)
from src.utils.logger import logger

class EEGProcessor:
    """Handles EEG data collection and processing."""
    
    def __init__(self, data_queue):
        """
        Initialize the EEG processor.
        
        Args:
            data_queue (queue.Queue): Queue for processed EEG data.
        """
        self.data_queue = data_queue
        self.board = None
        self.debug_mode = False
        self.simulated_sampling_rate = SIMULATED_SAMPLING_RATE
        self.collection_start_time = None
        self.collection_thread = None
        self.running = False
        
        # Data buffer settings - will be initialized when connecting to MUSE
        self.data_buffer = None
        self.buffer_size = None
        self.buffer_position = 0
        self.eeg_channels = None
    
    def connect_muse(self, debug=False):
        """
        Connect to MUSE device or enable debug mode.
        
        Args:
            debug (bool): If True, use simulated data instead of real device.
            
        Returns:
            bool: True if connection successful, False otherwise.
        """
        self.debug_mode = debug
        
        if debug:
            logger.info("Running in debug mode - simulating MUSE connection")
            return True
            
        # Real MUSE connection attempt
        try:
            params = BrainFlowInputParams()
            params.serial_port = ""  # Auto-discovery
            params.timeout = 15
            
            # Use the correct board ID for Muse 2 (38)
            board_id = 38  # Muse 2 board ID constant from BrainFlow
            BoardShim.enable_dev_board_logger()
            
            self.board = BoardShim(board_id, params)
            self.board.prepare_session()
            if not self.board.is_prepared():
                raise ConnectionError("Board preparation failed")
            
            # Initialize the data buffer for all channels
            sampling_rate = BoardShim.get_sampling_rate(board_id)
            total_channels = self.board.get_num_rows(board_id)  # Get total number of channels
            logger.info(f"Total channels from MUSE: {total_channels}")
            
            # Buffer 5 seconds of data (adjust if needed)
            self.buffer_size = int(5 * sampling_rate)
            self.data_buffer = np.zeros((total_channels, self.buffer_size))
            self.buffer_position = 0
            
            # Store channel information
            self.eeg_channels = BoardShim.get_eeg_channels(board_id)
            logger.info(f"EEG channels: {self.eeg_channels}")
            
            self.board.start_stream()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MUSE: {e}")
            return False
    
    def is_connected(self):
        """
        Check if connected to the EEG device.
        
        Returns:
            bool: True if connected or in debug mode, False otherwise.
        """
        if self.debug_mode:
            return True
        return self.board is not None and self.board.is_prepared()
    
    def start_data_collection(self):
        """Start the EEG data collection thread."""
        if self.collection_thread is not None and self.collection_thread.is_alive():
            logger.warning("Data collection already running")
            return
        
        logger.info("Starting data collection thread...")
        self.running = True
        self.collection_start_time = time.time()
        self.collection_thread = threading.Thread(target=self._data_collection_worker, daemon=True)
        self.collection_thread.start()
        logger.info("Data collection thread started")
    
    def _data_collection_worker(self):
        """Continuously collect EEG data."""
        try:
            logger.info("Starting data collection worker...")
            
            if self.debug_mode:
                sampling_rate = self.simulated_sampling_rate
                logger.info("Running in debug mode with simulated data")
            else:
                sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
            
            logger.info(f"Sampling rate: {sampling_rate} Hz")
            
            # Calculate update interval based on window size
            # We use a sliding window with 25% overlap for smooth updates
            update_interval = WINDOW_SIZE * 0.25  # 25% overlap between consecutive windows
            logger.info(f"Update interval: {update_interval:.2f}s")
            last_update = time.time()
            
            while self.running:
                try:
                    current_time = time.time()
                    if current_time - last_update >= update_interval:
                        result = self._get_band_powers()
                        if result is not None:
                            center_time, powers = result
                            logger.debug(f"Putting data in queue at center_time={center_time:.2f}")
                            self.data_queue.put((center_time, powers))
                            last_update = current_time
                        else:
                            logger.warning("No valid power values returned")
                            time.sleep(0.1)  # Add delay on error to prevent spam
                    else:
                        # Small sleep to prevent CPU overload but maintain responsiveness
                        time.sleep(0.001)
                except Exception as e:
                    logger.error(f"Error in data collection loop: {e}")
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Fatal error in data collection worker: {e}", exc_info=True)
    
    def _get_band_powers(self):
        """
        Calculate power for each frequency band.
        
        Returns:
            tuple: (center_time, powers_dict) or None if error
        """
        try:
            if self.debug_mode:
                return self._get_simulated_band_powers()
                
            logger.info("Getting board data...")
            
            # Get new data from the board
            new_data = self.board.get_current_board_data(12)  # Get the latest 12 samples
            if new_data.size == 0:
                logger.warning("No new data received from board")
                return None
            
            logger.debug(f"Got new data with shape: {new_data.shape}")
            
            # Update the circular buffer with new data
            new_samples = new_data.shape[1]
            if new_samples > 0:
                # Calculate where to insert the new data
                end_pos = (self.buffer_position + new_samples) % self.buffer_size
                if end_pos > self.buffer_position:
                    # Simple case: no wrapping around buffer end
                    self.data_buffer[:, self.buffer_position:end_pos] = new_data
                else:
                    # Data wraps around buffer end
                    first_chunk = self.buffer_size - self.buffer_position
                    self.data_buffer[:, self.buffer_position:] = new_data[:, :first_chunk]
                    self.data_buffer[:, :end_pos] = new_data[:, first_chunk:]
                
                self.buffer_position = end_pos
            
            # Calculate window size in samples
            sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
            window_samples = int(WINDOW_SIZE * sampling_rate)
            
            # Get the most recent window of data from the buffer
            if self.buffer_position >= window_samples:
                window_data = self.data_buffer[:, self.buffer_position - window_samples:self.buffer_position]
            else:
                # Handle wrap-around case
                first_part = self.data_buffer[:, self.buffer_position - window_samples:]
                second_part = self.data_buffer[:, :self.buffer_position]
                window_data = np.concatenate((first_part, second_part), axis=1)
            
            # Calculate the timestamp at the center of the window
            current_time = time.time()
            window_center_offset = WINDOW_SIZE / 2
            center_time = current_time - window_center_offset
            
            if not hasattr(self, 'collection_start_time'):
                logger.warning("Collection start time not set, initializing now")
                self.collection_start_time = time.time()
            
            center_time_relative = center_time - self.collection_start_time
            
            band_powers = {band: [] for band in FREQ_BANDS.keys()}
            
            # Process only EEG channels
            for ch_idx, ch in enumerate(self.eeg_channels):
                channel_data = window_data[ch]
                if len(channel_data) == 0:
                    logger.warning(f"No data for channel {ch}")
                    continue
                
                try:
                    # Apply notch filter to remove line noise (50/60 Hz)
                    notch_data = channel_data.copy()
                    DataFilter.remove_environmental_noise(
                        notch_data,
                        sampling_rate,
                        FilterTypes.BUTTERWORTH.value
                    )
                    
                    for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                        # Use the notch-filtered data for bandpass
                        band_data = notch_data.copy()
                        DataFilter.perform_bandpass(
                            band_data,
                            sampling_rate,
                            low_freq,
                            high_freq,
                            4,
                            FilterTypes.BUTTERWORTH.value,
                            0
                        )
                        power = np.mean(np.abs(band_data))
                        if not np.isnan(power):
                            band_powers[band_name].append(power)
                        else:
                            logger.warning(f"NaN power value for {band_name} in channel {ch}")
                except Exception as e:
                    logger.error(f"Error processing channel {ch}: {e}")
                    continue
            
            powers_result = {band: np.mean(powers) if powers else 0 
                           for band, powers in band_powers.items()}
            
            logger.info(f"Calculated powers: {', '.join([f'{k}:{v:.2e}' for k, v in powers_result.items()])}")
            return center_time_relative, powers_result
            
        except Exception as e:
            logger.error(f"Error in get_band_powers: {e}", exc_info=True)
            if self.debug_mode:
                return self._get_simulated_band_powers()
            return None
    
    def _get_simulated_band_powers(self):
        """
        Generate simulated EEG data for testing.
        
        Returns:
            tuple: (center_time, powers_dict)
        """
        try:
            if not hasattr(self, 'collection_start_time') or self.collection_start_time is None:
                self.collection_start_time = time.time()
            
            current_time = time.time()
            center_time = current_time - WINDOW_SIZE / 2
            center_time_relative = center_time - self.collection_start_time
            
            # Generate simulated power values
            t = center_time_relative
            powers = {
                'delta': 100 * (1 + 0.5 * np.sin(0.1 * t)),
                'theta': 50 * (1 + 0.3 * np.sin(0.2 * t)),
                'alpha': 25 * (1 + 0.8 * np.sin(0.5 * t)),
                'beta': 10 * (1 + 0.4 * np.sin(0.8 * t)),
                'gamma': 5 * (1 + 0.2 * np.sin(1.0 * t))
            }
            
            logger.debug(f"Generated simulated data at t={center_time_relative:.2f}s")
            return center_time_relative, powers
            
        except Exception as e:
            logger.error(f"Error generating simulated data: {e}", exc_info=True)
            return None
    
    def get_signal_quality(self):
        """
        Get signal quality data from the EEG channels.
        
        Returns:
            numpy.ndarray: EEG data or None if in debug mode or error.
        """
        if self.debug_mode:
            return None
            
        try:
            return self.board.get_board_data()
        except Exception as e:
            logger.error(f"Error getting signal quality data: {e}")
            return None
    
    def stop(self):
        """Stop data collection and release resources."""
        logger.info("Stopping EEG processor...")
        self.running = False
        
        if self.collection_thread is not None:
            self.collection_thread.join(timeout=1.0)
            self.collection_thread = None
        
        if not self.debug_mode and self.board is not None:
            try:
                logger.debug("Stopping MUSE board stream")
                self.board.stop_stream()
                self.board.release_session()
                self.board = None
            except Exception as e:
                logger.error(f"Error stopping MUSE board: {e}") 