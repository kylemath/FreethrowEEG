from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Set up parameters for Muse
params = BrainFlowInputParams()
params.serial_port = ""  # Leave blank for auto-discovery

# Use the Muse BLED board ID
BoardShim.enable_dev_board_logger()
muse_board_id = 38  # For Muse 2 & Muse S

# Connect to Muse headband
try:
    board = BoardShim(muse_board_id, params)
    board.prepare_session()
    print("✅ Muse headband found and connected!")
    board.start_stream()
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Initialize webcam with better error handling
def initialize_webcam():
    print("Attempting to connect to webcam...")
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        # Wait a bit to make sure the camera has time to initialize
        time.sleep(1)
        
        if cap.isOpened():
            # Try to read a test frame
            ret, frame = cap.read()
            if ret:
                print(f"✅ Webcam connected and working on index {camera_index}!")
                return cap
            cap.release()
    
    print("\n❌ Could not access webcam. Please check the following:")
    print("1. On macOS, go to System Settings > Privacy & Security > Camera")
    print("2. Ensure your Terminal/IDE has permission to access the camera")
    print("3. Make sure no other application is using the camera\n")
    board.stop_stream()
    board.release_session()
    exit()

# Initialize webcam
cap = initialize_webcam()

# Define frequency bands
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# Replace the get_alpha_power function with this new function
def get_band_powers():
    """Retrieves EEG data and extracts power for all frequency bands."""
    data = board.get_board_data()
    eeg_channels = BoardShim.get_eeg_channels(muse_board_id)
    sampling_rate = BoardShim.get_sampling_rate(muse_board_id)
    
    band_powers = {band: [] for band in FREQ_BANDS.keys()}
    
    for ch in eeg_channels:
        channel_data = data[ch]
        
        # Calculate power for each frequency band
        for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
            # Create a copy of the data for this band
            band_data = channel_data.copy()
            
            # Apply bandpass filter for the specific frequency band
            DataFilter.perform_bandpass(
                band_data,
                sampling_rate,
                low_freq,
                high_freq,
                4,
                FilterTypes.BUTTERWORTH.value,
                0
            )
            
            # Calculate power
            power = np.mean(np.abs(band_data))
            band_powers[band_name].append(power)
    
    # Average across channels for each band
    return {band: np.mean(powers) for band, powers in band_powers.items()}

# Set up plotting
plt.ion()  # Enable interactive plotting
fig = plt.figure(figsize=(15, 6))
eeg_ax = fig.add_subplot(121)  # Left subplot for EEG
webcam_ax = fig.add_subplot(122)  # Right subplot for webcam

lines = {}
colors = {'delta': 'blue', 'theta': 'green', 'alpha': 'red', 
          'beta': 'purple', 'gamma': 'orange'}

# Initialize empty data for plotting
time_points = []
band_data = {band: [] for band in FREQ_BANDS.keys()}

# Create a line for each frequency band
for band in FREQ_BANDS.keys():
    line, = eeg_ax.plot([], [], label=band, color=colors[band])
    lines[band] = line

eeg_ax.set_xlabel('Time (s)')
eeg_ax.set_ylabel('Power')
eeg_ax.set_title('EEG Frequency Band Powers')
eeg_ax.legend()
eeg_ax.grid(True)

webcam_ax.set_title('Webcam Feed')
webcam_ax.axis('off')

# Modified main loop
start_time = time.time()
try:
    while True:
        # Get EEG data
        powers = get_band_powers()
        current_time = time.time() - start_time
        time_points.append(current_time)
        
        # Update EEG plot
        for band in FREQ_BANDS.keys():
            band_data[band].append(powers[band])
            lines[band].set_data(time_points, band_data[band])
        
        # Adjust EEG plot limits
        eeg_ax.set_xlim(max(0, current_time - 30), current_time + 2)
        eeg_ax.set_ylim(0, max(max(data) for data in band_data.values()) * 1.1)
        
        # Get and display webcam frame
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Clear previous image
            webcam_ax.clear()
            webcam_ax.imshow(frame_rgb)
            webcam_ax.axis('off')
            webcam_ax.set_title('Webcam Feed')
        
        plt.pause(0.1)  # Update plot
        
except KeyboardInterrupt:
    print("Stopping streams...")
    board.stop_stream()
    board.release_session()
    cap.release()
    plt.ioff()
    plt.show()
