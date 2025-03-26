from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
import sounddevice as sd  # For audio cues
import threading
import sys
from pathlib import Path
import queue
import collections
from matplotlib.animation import FuncAnimation
import pywt  # For wavelet denoising
from sklearn.decomposition import FastICA  # For ICA-based artifact removal
from session_manager import Session

# Signal quality thresholds
SIGNAL_THRESHOLD = 500  # Maximum acceptable signal value
MIN_SIGNAL_QUALITY = 0.7  # Minimum proportion of good data points

class SignalProcessor:
    def __init__(self):
        self.adaptive_thresholds = AdaptiveThresholds()
        self.noise_reduction = PhaseSpecificNoiseReduction()
        self.board = None
        self.noise_reduction.set_signal_processor(self)
        
    def set_board(self, board):
        """Set the board reference for sampling rate access"""
        self.board = board
        self.noise_reduction.set_board(board)
        
    def apply_wavelet_denoising(self, data):
        """Apply wavelet denoising with soft thresholding"""
        wavelet = 'db4'  # Daubechies 4 wavelet
        level = 4  # Decomposition level
        
        # Perform wavelet denoising
        coeffs = pywt.wavedec(data, wavelet, level=level)
        # Apply soft thresholding
        threshold = np.median(np.abs(coeffs[-1])) / 0.6745
        coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        # Reconstruct signal
        return pywt.waverec(coeffs_thresholded, wavelet)
    
    def detect_emg_artifacts(self, data, threshold=0.75):
        """Detect EMG artifacts using high-frequency power"""
        if not self.board:
            return np.zeros_like(data, dtype=bool)
            
        # Compute high frequency power (>30 Hz)
        sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
        high_freq_data = data.copy()
        DataFilter.perform_bandpass(
            high_freq_data,
            sampling_rate,
            30,
            50,
            4,
            FilterTypes.BUTTERWORTH.value,
            0
        )
        power = np.abs(high_freq_data)
        return power > (threshold * np.median(power))
    
    def apply_ica_cleaning(self, data, emg_mask):
        """Apply ICA-based artifact removal"""
        if not self.board:
            return data
            
        sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
        if len(data) < 2 * sampling_rate:  # Need enough data points
            return data
            
        # Apply ICA
        ica = FastICA(n_components=4)  # 4 components for 4 channels
        components = ica.fit_transform(data.reshape(-1, 1))
        
        # Identify artifact components (high correlation with EMG mask)
        artifact_idx = []
        for i in range(components.shape[1]):
            corr = np.corrcoef(components[:, i], emg_mask)[0, 1]
            if abs(corr) > 0.3:  # Correlation threshold
                artifact_idx.append(i)
                
        # Reconstruct without artifact components
        clean_data = data.copy()
        if artifact_idx:
            components[:, artifact_idx] = 0
            clean_data = ica.inverse_transform(components)
            
        return clean_data.flatten()

class AdaptiveThresholds:
    def __init__(self):
        self.baseline_beta = None
        self.adjustment_rate = 0.1
        
    def update_thresholds(self, new_data):
        if self.baseline_beta is None:
            self.baseline_beta = np.median(new_data)
        else:
            # Adaptive adjustment
            self.baseline_beta = (1 - self.adjustment_rate) * self.baseline_beta + \
                               self.adjustment_rate * np.median(new_data)
        return self.baseline_beta

class PhaseSpecificNoiseReduction:
    def __init__(self):
        self.pre_shot_threshold = 0.6  # More permissive
        self.shot_threshold = 0.8      # Stricter
        self.post_threshold = 0.7      # Balanced
        self.board = None
        self.signal_processor = None
        
    def set_board(self, board):
        """Set the board reference for sampling rate access"""
        self.board = board
        
    def set_signal_processor(self, signal_processor):
        """Set reference to signal processor for access to EMG detection"""
        self.signal_processor = signal_processor
        
    def process_phase(self, data, phase):
        if not self.board:
            return data
            
        if phase == 'pre_shot':
            # Prioritize preserving brain signals
            return self._conservative_cleaning(data)
        elif phase == 'shot':
            # Aggressive EMG rejection
            return self._aggressive_cleaning(data)
        else:
            # Balanced approach
            return self._standard_cleaning(data)
            
    def _conservative_cleaning(self, data):
        """Gentle cleaning to preserve brain signals"""
        # Apply minimal filtering
        filtered = data.copy()
        sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
        DataFilter.perform_bandpass(
            filtered,
            sampling_rate,
            1,
            50,
            2,  # Lower order
            FilterTypes.BUTTERWORTH.value,
            0
        )
        return filtered
        
    def _aggressive_cleaning(self, data):
        """Strong artifact rejection"""
        if not self.signal_processor:
            return self._standard_cleaning(data)
            
        # Apply more aggressive filtering
        filtered = data.copy()
        sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
        DataFilter.perform_bandpass(
            filtered,
            sampling_rate,
            1,
            50,
            6,  # Higher order
            FilterTypes.BUTTERWORTH.value,
            0
        )
        # Additional EMG rejection
        emg_mask = self.signal_processor.detect_emg_artifacts(filtered, self.shot_threshold)
        return self.signal_processor.apply_ica_cleaning(filtered, emg_mask)
        
    def _standard_cleaning(self, data):
        """Balanced approach"""
        filtered = data.copy()
        sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
        DataFilter.perform_bandpass(
            filtered,
            sampling_rate,
            1,
            50,
            4,  # Standard order
            FilterTypes.BUTTERWORTH.value,
            0
        )
        return filtered

class FreethrowUI:
    def __init__(self):
        # Constants
        self.FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}
        self.BUFFER_LENGTH = 30000  # Up to ~50 minutes at 10Hz
        self.PLOT_UPDATE_INTERVAL = 0.1
        self.SIGNAL_THRESHOLD = 500
        self.MIN_SIGNAL_QUALITY = 0.7
        
        # Line noise frequency (Hz) - typically 50Hz in Europe/Asia, 60Hz in Americas
        self.LINE_NOISE_FREQ = 60  # Change to 50 if in Europe/Asia
        
        # Shot timing constants
        self.PREP_TIME = 5
        self.PRE_SHOT_DURATION = 5
        self.SHOT_DURATION = 2
        self.POST_SHOT_DURATION = 3
        
        # Video settings
        self.FRAME_WIDTH = 640
        self.FRAME_HEIGHT = 480
        self.FPS = 30
        
        # Audio settings
        self.SAMPLE_RATE = 44100
        self.BEEP_DURATION = 0.1
        
        # Initialize state variables
        self.setup_complete = False
        self.player_id = None
        self.num_shots = None
        self.board = None
        self.cap = None
        self.current_shot = 0
        self.recording_shot = False
        self.shot_phase = "ready"
        self.phase_start_time = None
        self.shot_start_time = None
        
        # Initialize data buffers
        self.data_queue = queue.Queue()
        self.eeg_buffer = {band: collections.deque(maxlen=self.BUFFER_LENGTH) 
                          for band in self.FREQ_BANDS.keys()}
        self.time_buffer = collections.deque(maxlen=self.BUFFER_LENGTH)
        
        # Generate beep sounds
        self.ready_beep = self.generate_beep(frequency=800)
        self.shot_beep = self.generate_beep(frequency=1200)
        
        # Add animation attribute
        self.animation = None
        
        # Add signal processing components
        self.signal_processor = SignalProcessor()
        
        # Start setup dialog
        self.show_setup_dialog()
    
    def generate_beep(self, frequency=1000, duration=None):
        """Generate a beep sound"""
        if duration is None:
            duration = self.BEEP_DURATION
        t = np.linspace(0, duration, int(self.SAMPLE_RATE * duration), False)
        return np.sin(2 * np.pi * frequency * t)
    
    def play_beep_nonblocking(self, beep):
        """Play a beep sound without blocking"""
        try:
            sd.play(beep, self.SAMPLE_RATE, blocking=False)
        except Exception as e:
            print(f"Error playing beep: {e}")
    
    def show_setup_dialog(self):
        """Show initial setup dialog"""
        self.setup_fig = plt.figure(figsize=(8, 6))
        self.setup_fig.canvas.manager.set_window_title('FreethrowEEG Setup')
        
        # Add text boxes for player info
        plt.figtext(0.1, 0.8, 'Player Name/ID:', size=10)
        self.player_box = TextBox(plt.axes([0.4, 0.8, 0.4, 0.05]), '', initial='001')
        
        plt.figtext(0.1, 0.7, 'Number of Shots:', size=10)
        self.shots_box = TextBox(plt.axes([0.4, 0.7, 0.4, 0.05]), '', initial='2')
        
        # Add status indicators and connect buttons
        self.muse_status = plt.figtext(0.1, 0.5, 'X MUSE Not Connected', color='red')
        self.muse_button = Button(plt.axes([0.6, 0.5, 0.2, 0.05]), 'Connect MUSE')
        self.muse_button.on_clicked(self.connect_muse)
        
        self.camera_status = plt.figtext(0.1, 0.4, 'X Camera Not Connected', color='red')
        self.camera_button = Button(plt.axes([0.6, 0.4, 0.2, 0.05]), 'Connect Camera')
        self.camera_button.on_clicked(self.connect_camera)
        
        self.start_button = Button(plt.axes([0.35, 0.2, 0.3, 0.1]), 'Start Session')
        self.start_button.on_clicked(self.start_session)
        self.start_button.set_active(False)  # Disable until setup complete
        
        plt.show()
    
    def connect_muse(self, event):
        """Connect to MUSE device"""
        self.muse_status.set_text('* Connecting to MUSE...')
        self.muse_status.set_color('orange')
        self.muse_button.set_active(False)
        self.muse_button.label.set_text('Connecting...')
        plt.pause(0.01)  # Force update
        
        try:
            params = BrainFlowInputParams()
            params.serial_port = ""  # Auto-discovery
            params.timeout = 15
            
            board_id = 38  # Muse 2 board ID
            BoardShim.enable_dev_board_logger()
            
            self.board = BoardShim(board_id, params)
            self.board.prepare_session()
            if not self.board.is_prepared():
                raise ConnectionError("Board preparation failed")
            self.board.start_stream()
            
            # Set board reference in signal processor
            self.signal_processor.set_board(self.board)
            
            self.muse_status.set_text('✓ MUSE Connected')
            self.muse_status.set_color('green')
            self.muse_button.label.set_text('Connected')
            self.check_ready()
            
        except Exception as e:
            self.muse_status.set_text(f'X MUSE Error: {str(e)[:20]}...')
            self.muse_status.set_color('red')
            self.muse_button.label.set_text('Retry Connect')
            self.muse_button.set_active(True)
        plt.pause(0.01)  # Force update
    
    def connect_camera(self, event):
        """Connect to camera"""
        self.camera_status.set_text('* Connecting to Camera...')
        self.camera_status.set_color('orange')
        self.camera_button.set_active(False)
        self.camera_button.label.set_text('Connecting...')
        plt.pause(0.01)  # Force update
        
        try:
            self.cap = cv2.VideoCapture(0)  # Try default camera first
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Could not read from camera")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, self.FPS)
            
            self.camera_status.set_text('✓ Camera Connected')
            self.camera_status.set_color('green')
            self.camera_button.label.set_text('Connected')
            self.check_ready()
            
        except Exception as e:
            self.camera_status.set_text(f'X Camera Error: {str(e)[:20]}...')
            self.camera_status.set_color('red')
            self.camera_button.label.set_text('Retry Connect')
            self.camera_button.set_active(True)
            if self.cap:
                self.cap.release()
        plt.pause(0.01)  # Force update
    
    def check_ready(self):
        """Check if all setup is complete"""
        player_ready = len(self.player_box.text.strip()) > 0
        shots_ready = self.shots_box.text.strip().isdigit()
        muse_ready = self.board is not None and self.board.is_prepared()
        camera_ready = self.cap is not None and self.cap.isOpened()
        
        self.start_button.set_active(player_ready and shots_ready and muse_ready and camera_ready)
        plt.pause(0.01)  # Force update
    
    def start_session(self, event):
        """Start the main session"""
        self.player_id = self.player_box.text.strip()  # Get exact ID as entered
        self.num_shots = int(self.shots_box.text.strip())
        self.setup_complete = True
        plt.close(self.setup_fig)
        self.show_main_interface()
        
        # Create session directory
        session_dir = Path("data") / self.player_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear any old data
        self.time_buffer.clear()
        for buffer in self.eeg_buffer.values():
            buffer.clear()
            
        # Start data collection
        self.start_data_collection()
        
        # Initial plot update
        plt.pause(0.1)
    
    def show_main_interface(self):
        """Show the main interface with EEG, video, and control buttons"""
        # Create figure before starting threads
        self.fig = plt.figure(figsize=(15, 8))
        self.fig.canvas.manager.set_window_title('FreethrowEEG Session')
        
        # Create subplots
        self.eeg_ax = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
        self.quality_ax = plt.subplot2grid((3, 3), (0, 2), rowspan=1)
        self.video_ax = plt.subplot2grid((3, 3), (1, 2), rowspan=1)
        
        # Add control buttons
        self.start_shot_button = Button(plt.axes([0.7, 0.15, 0.2, 0.08]), 'Start Shot')
        self.start_shot_button.on_clicked(self.start_shot)
        
        self.success_button = Button(plt.axes([0.7, 0.05, 0.1, 0.08]), 'Made ✓')
        self.success_button.on_clicked(lambda x: self.mark_shot(True))
        self.success_button.set_active(False)
        
        self.fail_button = Button(plt.axes([0.8, 0.05, 0.1, 0.08]), 'Miss ✗')
        self.fail_button.on_clicked(lambda x: self.mark_shot(False))
        self.fail_button.set_active(False)
        
        # Add status text
        self.status_text = plt.figtext(0.1, 0.05, 'Ready to start', size=10)
        
        # Initialize plots
        self.setup_plots()
        
        # Start data collection
        self.start_data_collection()
        
        # Create animation with save_count to prevent warning
        self.animation = FuncAnimation(self.fig, self.animate, interval=100, 
                                     blit=True, save_count=100)
        
        # Show the figure
        plt.show()
    
    def setup_plots(self):
        """Initialize all plots"""
        # Set up EEG plot
        self.lines = {}
        self.colors = {'delta': 'blue', 'theta': 'green', 'alpha': 'red',
                      'beta': 'purple', 'gamma': 'orange'}
        
        for band in self.FREQ_BANDS.keys():
            line, = self.eeg_ax.plot([], [], label=band, color=self.colors[band])
            self.lines[band] = line
        
        # Set up log scale for power values
        self.eeg_ax.set_yscale('log')
        
        # Configure tick marks to be more readable
        self.eeg_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1e}'))
        
        # Create a twin x-axis for elapsed time
        self.elapsed_ax = self.eeg_ax.twinx()
        self.elapsed_ax.spines['right'].set_position(('outward', 60))  # Move it outward for clarity
        self.elapsed_ax.yaxis.set_visible(False)  # Hide y-axis of twin
        
        self.eeg_ax.set_xlabel('Time (s)')
        self.eeg_ax.set_ylabel('Power (log scale)')
        self.eeg_ax.set_title('EEG Frequency Band Powers')
        self.eeg_ax.legend()
        self.eeg_ax.grid(True)
        self.eeg_ax.set_xlim(0, 10)
        self.eeg_ax.set_ylim(0.1, 1000)  # Set reasonable log scale limits
        
        # Set up quality plot
        self.setup_quality_plot()
        
        # Set up video plot
        self.video_ax.set_title('Webcam Feed')
        self.video_ax.axis('off')
        # Initialize video image
        self.video_image = self.video_ax.imshow(np.zeros((self.FRAME_HEIGHT, self.FRAME_WIDTH, 3)))
        
        # Store all artists that need to be updated
        self.artists = list(self.lines.values())
        self.artists.extend(self.quality_circles.values())
        self.artists.append(self.video_image)
        # Add axes to the list of artists to ensure ticks update
        self.artists.extend([self.eeg_ax, self.elapsed_ax])
    
    def setup_quality_plot(self):
        """Set up the signal quality visualization"""
        self.MUSE_ELECTRODE_POSITIONS = {
            'TP9': (-0.8, 0),    # Left ear
            'AF7': (-0.4, 0.8),  # Left forehead
            'AF8': (0.4, 0.8),   # Right forehead
            'TP10': (0.8, 0),    # Right ear
        }
        
        self.quality_ax.clear()
        # Draw head outline
        circle = plt.Circle((0, 0.3), 0.8, fill=False, color='gray')
        self.quality_ax.add_patch(circle)
        # Draw nose
        self.quality_ax.plot([0, 0], [1.1, 0.9], 'gray')
        
        # Create circles for each electrode
        self.quality_circles = {}
        self.quality_text = {}
        for name, pos in self.MUSE_ELECTRODE_POSITIONS.items():
            circle = plt.Circle(pos, 0.1, color='gray', alpha=0.5)
            text = self.quality_ax.text(pos[0], pos[1]-0.2, name, ha='center', va='center')
            self.quality_ax.add_patch(circle)
            self.quality_circles[name] = circle
            self.quality_text[name] = text
        
        self.quality_ax.set_xlim(-1.2, 1.2)
        self.quality_ax.set_ylim(-0.5, 1.5)
        self.quality_ax.axis('off')
        self.quality_ax.set_title('Signal Quality')
    
    def update_signal_quality(self, data, eeg_channels):
        """Update signal quality visualization"""
        for i, channel in enumerate(eeg_channels):
            if i >= len(self.MUSE_ELECTRODE_POSITIONS):
                break
            
            electrode = list(self.MUSE_ELECTRODE_POSITIONS.keys())[i]
            channel_data = data[channel]
            
            if len(channel_data) > 0:
                has_nans = np.isnan(channel_data).any()
                above_threshold = np.abs(channel_data) > self.SIGNAL_THRESHOLD
                quality = 0 if has_nans else np.sum(~above_threshold) / len(channel_data)
            else:
                quality = 0
            
            if quality > 0.8:
                color, alpha = 'green', 0.8
            elif quality > 0.5:
                color, alpha = 'yellow', 0.6
            else:
                color, alpha = 'red', 0.4
            
            circle = self.quality_circles[electrode]
            circle.set_color(color)
            circle.set_alpha(alpha)
            # Make sure the circle is visible
            if not circle.get_visible():
                circle.set_visible(True)
    
    def get_band_powers(self):
        """Calculate power for each frequency band with enhanced noise reduction"""
        data = self.board.get_board_data()
        if data.size == 0:
            return None
            
        eeg_channels = BoardShim.get_eeg_channels(self.board.board_id)
        sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
        
        # Process shorter chunks of data to maintain responsiveness
        # Get last 1 second of data (sampling_rate points)
        data = data[:, -min(sampling_rate, data.shape[1]):]
        
        band_powers = {band: [] for band in self.FREQ_BANDS.keys()}
        
        for ch in eeg_channels:
            channel_data = data[ch]
            if len(channel_data) == 0:
                continue
            
            try:
                # Enhanced noise reduction based on current phase
                if self.shot_phase == "ready":
                    cleaned_data = self.signal_processor.noise_reduction.process_phase(
                        channel_data, 'pre_shot')
                elif self.shot_phase == "recording":
                    cleaned_data = self.signal_processor.noise_reduction.process_phase(
                        channel_data, 'shot')
                else:
                    cleaned_data = self.signal_processor.noise_reduction.process_phase(
                        channel_data, 'post')
                
                # Apply notch filter to remove line noise (50/60 Hz)
                notch_data = cleaned_data.copy()
                DataFilter.remove_environmental_noise(
                    notch_data,
                    sampling_rate,
                    FilterTypes.BUTTERWORTH.value
                )
                    
                for band_name, (low_freq, high_freq) in self.FREQ_BANDS.items():
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
                        
                # Update adaptive thresholds
                if band_name == 'beta':
                    self.signal_processor.adaptive_thresholds.update_thresholds(power)
                    
            except Exception as e:
                print(f"Error processing channel {ch}: {e}")
                continue
        
        # Only compute mean if we have data
        powers = {band: np.mean(powers) if powers else 0 
                 for band, powers in band_powers.items()}
        
        # Add tension feedback
        if 'beta' in powers and self.signal_processor.adaptive_thresholds.baseline_beta:
            tension_color = self.get_tension_feedback(
                powers['beta'],
                self.signal_processor.adaptive_thresholds.baseline_beta
            )
            self.update_tension_indicator(tension_color)
            
        return powers
    
    def get_tension_feedback(self, beta_power, baseline):
        """Get tension feedback color based on beta power"""
        ratio = beta_power / baseline
        if ratio > 1.5:
            return 'red'      # Too tense
        elif ratio > 1.2:
            return 'yellow'   # Slightly elevated
        else:
            return 'green'    # Good
            
    def update_tension_indicator(self, color):
        """Update the tension indicator in the UI"""
        if hasattr(self, 'tension_indicator'):
            self.tension_indicator.set_color(color)
        else:
            # Create tension indicator if it doesn't exist
            self.tension_indicator = plt.Circle((0.95, 0.95), 0.02, 
                                             color=color, transform=self.fig.transFigure)
            self.fig.add_artist(self.tension_indicator)
    
    def data_collection_worker(self):
        """Continuously collect EEG data"""
        collection_start_time = time.time()
        sampling_rate = BoardShim.get_sampling_rate(self.board.board_id)
        print(f"EEG sampling rate: {sampling_rate} Hz")
        
        update_interval = 0.05  # 20Hz update rate for smoother display
        last_update = time.time()
        
        while True:
            try:
                current_time = time.time()
                if current_time - last_update >= update_interval:
                    powers = self.get_band_powers()
                    if powers is not None:
                        self.data_queue.put((current_time - collection_start_time, powers))
                        last_update = current_time
                else:
                    # Small sleep to prevent CPU overload but maintain responsiveness
                    time.sleep(0.001)
            except Exception as e:
                print(f"Error in data collection: {e}")
                time.sleep(0.1)
    
    def start_data_collection(self):
        """Start data collection and plot update threads"""
        # Start data collection thread
        self.data_thread = threading.Thread(target=self.data_collection_worker, daemon=True)
        self.data_thread.start()
        
        # Start plot update thread
        self.update_thread = threading.Thread(target=self.update_plots, daemon=True)
        self.update_thread.start()
        
        # Wait a moment for initial data
        time.sleep(0.5)
    
    def update_plots(self):
        """Update plots continuously"""
        while True:
            try:
                # Process any new data
                while not self.data_queue.empty():
                    current_time, powers = self.data_queue.get()
                    self.time_buffer.append(current_time)
                    for band, power in powers.items():
                        self.eeg_buffer[band].append(power)
                
                # Small sleep to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                print(f"Error in plot loop: {e}")
                time.sleep(0.1)
    
    def animate(self, frame):
        """Animation function for matplotlib"""
        try:
            # Update EEG plot
            for band in self.FREQ_BANDS.keys():
                times = list(self.time_buffer)
                powers = list(self.eeg_buffer[band])
                if len(times) > 1:  # Need at least 2 points to interpolate
                    # Ensure times are strictly increasing
                    times, unique_idx = np.unique(times, return_index=True)
                    powers = np.array(powers)[unique_idx]
                    self.lines[band].set_data(times, powers)
            
            # Update signal quality
            eeg_data = self.board.get_board_data()
            if eeg_data.size > 0:
                self.update_signal_quality(eeg_data, 
                                        BoardShim.get_eeg_channels(self.board.board_id))
            
            # Update video frame
            ret, frame = self.cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_image.set_array(frame_rgb)
            
            # Adjust EEG plot limits to show all data from 0 to current time
            if len(self.time_buffer) > 0:
                # Get current time
                current_time = list(self.time_buffer)[-1] if self.time_buffer else 0
                
                # Always show from 0 to current time with a small padding
                padding = max(5, current_time * 0.05)  # At least 5 seconds or 5% of total time
                xlim = (0, current_time + padding)
                self.eeg_ax.set_xlim(xlim)
                self.elapsed_ax.set_xlim(xlim)
                
                # Generate appropriate tick marks - distribute evenly
                if current_time > 0:
                    # Calculate appropriate tick interval based on the time span
                    if current_time < 30:
                        tick_interval = 5  # 5-second intervals for shorter recordings
                    elif current_time < 120:
                        tick_interval = 15  # 15-second intervals for 30s-2min recordings
                    elif current_time < 300:
                        tick_interval = 30  # 30-second intervals for 2-5min recordings
                    else:
                        tick_interval = 60  # 1-minute intervals for longer recordings
                    
                    # Create evenly spaced ticks
                    max_ticks = 10  # Limit number of ticks for readability
                    ticks = np.linspace(0, current_time, min(max_ticks, int(current_time/tick_interval) + 1))
                    
                    # Update main axis (left)
                    self.eeg_ax.set_xticks(ticks)
                    if current_time > 60:  # More than 1 minute
                        self.eeg_ax.set_xticklabels([f"{int(t/60)}:{int(t%60):02d}" for t in ticks])
                        self.eeg_ax.set_xlabel('Time (min:sec)')
                    else:
                        self.eeg_ax.set_xticklabels([f"{int(t)}" for t in ticks])
                        self.eeg_ax.set_xlabel('Time (sec)')
                    
                    # Update elapsed time axis (right)
                    self.elapsed_ax.set_xticks(ticks)
                    self.elapsed_ax.set_xticklabels([f"{int(t)}" for t in ticks])
                    self.elapsed_ax.set_xlabel('Elapsed Time (sec)', labelpad=10)
                    
                    # Force update of tick labels
                    self.eeg_ax.figure.canvas.draw()
                    self.elapsed_ax.figure.canvas.draw()
            
            # Return all artists that need to be redrawn
            return self.artists
            
        except Exception as e:
            print(f"Error in animation update: {e}")
            return []
    
    def start_shot(self, event):
        """Start the shot sequence"""
        if self.current_shot >= self.num_shots:
            self.status_text.set_text('All shots completed!')
            return
        
        self.shot_phase = "prep"
        self.phase_start_time = time.time()
        self.start_shot_button.set_active(False)
        self.status_text.set_text('Get ready...')
        
        # Start shot sequence timer
        self.shot_timer = threading.Timer(self.PREP_TIME, self.start_recording)
        self.shot_timer.start()
        
        # Schedule ready beep
        threading.Timer(self.PREP_TIME - 2, 
                      lambda: self.play_beep_nonblocking(self.ready_beep)).start()
    
    def start_recording(self):
        """Start recording the shot"""
        if self.shot_phase != "prep":
            return
        
        self.shot_phase = "recording"
        self.recording_shot = True
        self.shot_start_time = time.time()
        self.status_text.set_text('SHOOT NOW!')
        self.play_beep_nonblocking(self.shot_beep)
        
        # Schedule end of recording after SHOT_DURATION + POST_SHOT_DURATION
        self.shot_timer = threading.Timer(self.SHOT_DURATION + self.POST_SHOT_DURATION, self.end_recording)
        self.shot_timer.start()
    
    def end_recording(self):
        """End the shot recording"""
        if self.shot_phase != "recording":
            return
        
        self.shot_phase = "review"
        self.recording_shot = False
        self.status_text.set_text('Was it good?')
        self.success_button.set_active(True)
        self.fail_button.set_active(True)
    
    def mark_shot(self, success):
        """Mark the shot as made or missed"""
        if self.shot_phase != "review":
            return
        
        self.current_shot += 1
        result = "MADE" if success else "MISSED"
        
        if self.current_shot >= self.num_shots:
            self.status_text.set_text('All shots completed! Saving data...')
            plt.pause(0.1)  # Force text update
            self.start_shot_button.set_active(False)
            # Save data first, then cleanup
            self.save_session_data(success)
            self.cleanup()
            plt.close('all')  # Close all windows immediately
        else:
            self.status_text.set_text(f'Shot {result} - Press Start Shot when ready')
            self.start_shot_button.set_active(True)
            # Save data for this shot
            self.save_session_data(success)
        
        self.shot_phase = "ready"
        self.success_button.set_active(False)
        self.fail_button.set_active(False)
    
    def save_session_data(self, success):
        """Save shot data to session"""
        try:
            # Create session if it doesn't exist
            if not hasattr(self, 'session'):
                self.session = Session(self.player_id, self.num_shots)
            
            # Get EEG data for this shot
            eeg_data = {
                'pre_shot': {band: list(self.eeg_buffer[band])[-500:-250] for band in self.FREQ_BANDS.keys()},
                'during_shot': {band: list(self.eeg_buffer[band])[-250:-100] for band in self.FREQ_BANDS.keys()},
                'post_shot': {band: list(self.eeg_buffer[band])[-100:] for band in self.FREQ_BANDS.keys()}
            }
            
            # Create shot data
            shot_data = {
                "shot_id": self.current_shot,
                "timestamp": time.time(),
                "success": success,
                "eeg_data": eeg_data,
                "video_path": str(self.session.session_dir / "session_recording.mp4")
            }
            
            # Add shot to session
            self.session.add_shot(shot_data)
            
        except Exception as e:
            print(f"Error saving shot data: {e}")
    
    def cleanup(self):
        """Clean up resources when session ends"""
        try:
            print("Starting cleanup...")
            if hasattr(self, 'animation'):
                self.animation.event_source.stop()
            
            if hasattr(self, 'data_thread'):
                self.data_thread.join(timeout=1.0)
                
            if hasattr(self, 'update_thread'):
                self.update_thread.join(timeout=1.0)
            
            if hasattr(self, 'shot_timer'):
                self.shot_timer.cancel()
            
            if self.board:
                print("Stopping EEG stream...")
                self.board.stop_stream()
                self.board.release_session()
                print("EEG stream stopped")
            
            if self.cap:
                print("Releasing camera...")
                self.cap.release()
                print("Camera released")
            
            print("Cleanup completed")
            
        except Exception as e:
            print(f"Error during cleanup: {e}")

# Start the application
if __name__ == "__main__":
    app = FreethrowUI()
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping application...")
    finally:
        app.cleanup()
        sys.exit(0)
