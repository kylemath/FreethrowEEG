from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes
import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from session_manager import initialize_session
import os
import sounddevice as sd  # For audio cues

# Initialize session first
session = initialize_session()

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
    
    # Try laptop camera first (usually index 0 or 1)
    for camera_index in [1, 0, 2]:  # Changed order to try index 1 first
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

# Create named window for keyboard focus
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

webcam_ax.set_title('Webcam Feed')
webcam_ax.axis('off')

# Audio cue settings
SAMPLE_RATE = 44100
BEEP_DURATION = 0.1  # seconds

def generate_beep(frequency=1000, duration=BEEP_DURATION):
    """Generate a beep sound."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = np.sin(2 * np.pi * frequency * t)
    return tone

# Generate beep sounds
ready_beep = generate_beep(frequency=800)  # Lower pitch for ready
shot_beep = generate_beep(frequency=1200)  # Higher pitch for shot

# Constants for shot timing
PREP_TIME = 5  # seconds to get ready
PRE_SHOT_DURATION = 5  # seconds to record before shot (increased from 3)
SHOT_DURATION = 2  # seconds to take the shot
POST_SHOT_DURATION = 3  # seconds to record after shot (increased from 2)

def play_beep(beep):
    """Play a beep sound."""
    sd.play(beep, SAMPLE_RATE)
    sd.wait()  # Wait for the sound to finish playing

# Modified draw_status function to include shot timing indicator
def draw_status(frame, status_text, recording=False, countdown=None):
    """Draw status text and recording indicator on frame"""
    height, width = frame.shape[:2]
    
    # Add semi-transparent overlay for text background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, height-60), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Add status text
    cv2.putText(frame, status_text, (10, height-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add countdown if provided
    if countdown is not None:
        # Make countdown red when it's close to shot time
        color = (0, 0, 255) if countdown <= 3 else (255, 255, 255)
        cv2.putText(frame, str(int(countdown)), (width//2 - 20, height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 3)
    
    # Add recording indicator if recording
    if recording:
        cv2.circle(frame, (width-30, height-30), 10, (0, 0, 255), -1)

# Modified main loop
start_time = time.time()
current_shot = 0
recording_shot = False
shot_start_time = None
pre_shot_data = None
during_shot_data = None

# Video settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')

# Create session video writer
session_video_path = str(session.session_dir / "session_recording.mp4")
video_writer = cv2.VideoWriter(session_video_path, FOURCC, FPS, (FRAME_WIDTH, FRAME_HEIGHT))

try:
    print("\nControls:")
    print("- Press SPACE to start each shot sequence")
    print("- You'll have 5 seconds to get ready")
    print("- Then 2 seconds to take your shot")
    print("- Press 'S' to mark the shot as successful")
    print("- Press 'F' to mark the shot as failed")
    print("- Press 'Q' to quit\n")
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    status_text = "Ready for shot 1 - Press SPACE when ready"
    shot_phase = "ready"  # ready, prep, recording, review
    phase_start_time = None
    
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
            countdown = None
            
            # Handle different phases
            if shot_phase == "prep" and phase_start_time:
                elapsed = time.time() - phase_start_time
                if elapsed < PREP_TIME:
                    countdown = PREP_TIME - elapsed
                    status_text = "Get ready..."
                    # Play ready beep at 3 seconds
                    if int(countdown) == 3 and countdown % 1 > 0.9:
                        play_beep(ready_beep)
                else:
                    shot_phase = "recording"
                    phase_start_time = time.time()
                    recording_shot = True
                    shot_start_time = current_time
                    pre_shot_index = max(0, len(time_points) - int(PRE_SHOT_DURATION * 10))
                    pre_shot_data = {
                        band: list(band_data[band][pre_shot_index:])
                        for band in FREQ_BANDS.keys()
                    }
                    print("\nRecording shot...")
                    play_beep(shot_beep)  # Play shot beep
            
            elif shot_phase == "recording" and phase_start_time:
                elapsed = time.time() - phase_start_time
                if elapsed < SHOT_DURATION:
                    countdown = SHOT_DURATION - elapsed
                    status_text = "SHOOT NOW!"
                else:
                    shot_phase = "post_recording"
                    phase_start_time = time.time()
                    recording_shot = False
                    during_shot_data = {
                        band: list(band_data[band])
                        for band in FREQ_BANDS.keys()
                    }
                    print("Recording post-shot data...")
            
            elif shot_phase == "post_recording" and phase_start_time:
                elapsed = time.time() - phase_start_time
                if elapsed < POST_SHOT_DURATION:
                    status_text = "Recording post-shot data..."
                else:
                    shot_phase = "review"
                    post_shot_data = {
                        band: list(band_data[band][-int(POST_SHOT_DURATION * 10):])
                        for band in FREQ_BANDS.keys()
                    }
                    status_text = "Was it good? Press 'S' for success, 'F' for failure"
                    print("Post-shot data captured. Press 'S' for success or 'F' for failure.")
            
            # Draw status and frame
            draw_status(frame, status_text, recording_shot, countdown)
            video_writer.write(frame)
            cv2.imshow('Camera Feed', frame)
            
            # Convert BGR to RGB for matplotlib display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            webcam_ax.clear()
            webcam_ax.imshow(frame_rgb)
            webcam_ax.axis('off')
            webcam_ax.set_title("Shot Recording" if recording_shot else "Webcam Feed")
        
        # Check for keyboard input
        key = cv2.waitKey(10) & 0xFF
        
        # Handle keyboard input
        if key == 32 and shot_phase == "ready":  # Spacebar to start sequence
            if current_shot >= session.num_shots:
                print("\nAll planned shots completed!")
                raise KeyboardInterrupt
            shot_phase = "prep"
            phase_start_time = time.time()
            print(f"\nStarting shot {current_shot + 1} sequence...")
            print("Get ready...")
        
        elif key in [ord('s'), ord('f')] and shot_phase == "review":  # Success/Failure
            success = (key == ord('s'))
            
            # Create shot data structure
            shot_data = {
                "shot_id": current_shot + 1,
                "timestamp": time.time(),
                "success": success,
                "eeg_data": {
                    "pre_shot": pre_shot_data,
                    "during_shot": during_shot_data,
                    "post_shot": post_shot_data
                },
                "video_timestamps": {
                    "start": shot_start_time,
                    "end": current_time
                },
                "duration": current_time - shot_start_time,
                "video_path": session_video_path
            }
            
            session.add_shot(shot_data)
            current_shot += 1
            result = "MADE" if success else "MISSED"
            print(f"Shot {current_shot} marked as {'successful' if success else 'failed'}!")
            
            if current_shot >= session.num_shots:
                print("\nAll planned shots completed!")
                raise KeyboardInterrupt
            
            # Reset for next shot
            shot_phase = "ready"
            phase_start_time = None
            pre_shot_data = None
            during_shot_data = None
            shot_start_time = None
            status_text = f"Shot {current_shot}: {result} - Press SPACE when ready for next shot"
        
        elif key == ord('q'):  # Quit
            raise KeyboardInterrupt
        
        plt.pause(0.01)  # Update plot frequently
        
except KeyboardInterrupt:
    print("\nStopping streams...")
    video_writer.release()
    board.stop_stream()
    board.release_session()
    cap.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows
    plt.close('all')  # Close all matplotlib windows
    print(f"\nSession completed with {current_shot} shots recorded.")
    print(f"Data saved in: {session.session_dir}")
    exit()  # Ensure the program exits completely
