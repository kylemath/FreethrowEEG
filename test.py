from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes
import numpy as np
import time

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


# Function to extract alpha power
def get_alpha_power():
    """Retrieves EEG data, filters it, and extracts alpha wave power."""
    data = board.get_board_data()
    eeg_channels = BoardShim.get_eeg_channels(muse_board_id)
    alpha_powers = []

    for ch in eeg_channels:
        # Perform bandpass filter to isolate alpha waves (8-12 Hz)
        DataFilter.perform_bandpass(data[ch], BoardShim.get_sampling_rate(muse_board_id), 8.0, 12.0, 4,
                                    FilterTypes.BUTTERWORTH.value, 0)
        alpha_power = np.mean(np.abs(data[ch]))  # Simple power estimation
        alpha_powers.append(alpha_power)

    return np.mean(alpha_powers)  # Average alpha power across channels


# Streaming and alpha wave extraction
try:
    while True:
        alpha_level = get_alpha_power()
        print(f"Alpha Power: {alpha_level:.4f}")
        time.sleep(1)  # Update every second
except KeyboardInterrupt:
    print("Stopping EEG stream...")
    board.stop_stream()
    board.release_session()
