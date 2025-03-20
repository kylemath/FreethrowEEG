"""
Configuration parameters for the FreethrowEEG application.
"""

# EEG processing constants
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}

# Buffer and timing constants
BUFFER_LENGTH = 30000  # Up to ~50 minutes at 10Hz
PLOT_UPDATE_INTERVAL = 0.1
SIGNAL_THRESHOLD = 500
MIN_SIGNAL_QUALITY = 0.7

# Debug settings
DEBUG_MODE_DEFAULT = False  # Default to real data mode
SIMULATED_SAMPLING_RATE = 256  # Hz

# Spectral analysis settings
WINDOW_SIZE = 1.0  # Window size for spectral analysis (in seconds)
                   # Lower for better time resolution, increase for better frequency resolution

# Line noise frequency (Hz) - typically 50Hz in Europe/Asia, 60Hz in Americas
LINE_NOISE_FREQ = 60  # Change to 50 if in Europe/Asia

# Shot timing constants
PREP_TIME = 5
PRE_SHOT_DURATION = 5
SHOT_DURATION = 2
POST_SHOT_DURATION = 3

# Video settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Audio settings
SAMPLE_RATE = 44100
BEEP_DURATION = 0.1

# MUSE electrode positions for visualization
MUSE_ELECTRODE_POSITIONS = {
    'TP9': (-0.8, 0),    # Left ear
    'AF7': (-0.4, 0.8),  # Left forehead
    'AF8': (0.4, 0.8),   # Right forehead
    'TP10': (0.8, 0),    # Right ear
} 