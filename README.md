# Freethrow EEG - Record EEG during freethrow shooting

This program records your brain activity while shooting free throws, helping you understand your mental state during successful and unsuccessful shots.

## Features

- Records EEG data from MUSE headband during free throw shooting
- Tracks multiple frequency bands (alpha, beta, theta, delta, gamma)
- Records shot success/failure
- Organizes data by player and session
- Video recording of shots
- Automated timing system with audio cues
- Structured data storage for easy analysis

## Prerequisites

1. Hardware Requirements:

   - MUSE EEG headband (charged and ready to pair)
   - Laptop with webcam
   - Basketball and hoop
   - Space to shoot free throws

2. Software Requirements:
   - Python 3.8 or higher
   - Bluetooth connectivity for MUSE headband
   - macOS, Linux, or Windows

## Setup Instructions

### First Time Setup

1. Clone the repository:

   ```bash
   git clone http://github.com/kylemath/FreethrowEEG
   cd FreethrowEEG
   ```

2. Create and activate virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Mac/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Before Each Practice Session

1. Ensure your MUSE headband is:

   - Fully charged
   - In pairing mode (light blinking)
   - Connected to your computer via Bluetooth

2. Find a suitable location:

   - Clear space for free throw shooting
   - Good lighting for video recording
   - Stable surface for your laptop
   - Position webcam to capture you and the basket

3. Activate virtual environment (if not already active):
   ```bash
   source .venv/bin/activate  # On Mac/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

### Running a Practice Session

1. Start the program:

   ```bash
   python3 test.py
   ```

2. Initial Setup:

   - Enter your name/ID when prompted
   - Enter the number of shots you plan to take

3. For Each Shot:

   - Press SPACEBAR when ready to start
   - Wait for the ready beep (5 seconds)
   - You'll get 5 seconds to prepare (countdown shown)
   - Take your shot within the 2-second window
   - Wait for 3 seconds of post-shot recording
   - Press 'S' if you made the shot
   - Press 'F' if you missed the shot

4. Session End:
   - Program automatically ends after all shots are completed
   - Or press 'Q' to quit early

### Data Storage

Your practice data is automatically organized in this structure:

```
data/
  └── [PlayerName]/              # Your personal folder
      └── [SessionDateTime]/     # Unique session folder
          ├── session_metadata.json  # Complete session data
          ├── shots.json            # Individual shot data
          └── session_recording.mp4 # Video recording
```

## Troubleshooting

1. If MUSE headband isn't connecting:

   - Ensure it's in pairing mode (light blinking)
   - Try removing and re-adding the device in your computer's Bluetooth settings
   - Make sure the headband is charged

2. If webcam isn't working:

   - Check if another application is using the camera
   - Try closing and reopening the program
   - Ensure proper lighting in the room

3. If program crashes:
   - Make sure all dependencies are installed correctly
   - Check that the MUSE headband is properly connected
   - Ensure you're using the correct Python version

For additional help or to report issues, please visit:
[GitHub Issues](http://github.com/kylemath/FreethrowEEG/issues)

## Technical Details

The program records:

- Full EEG spectrum (delta, theta, alpha, beta, gamma waves)
- Shot success/failure
- Timing of each shot
- Video feed (when enabled)

### EEG Processing

- EEG data is collected continuously from the MUSE headband
- Frequency bands are extracted using bandpass filtering techniques
- A 2.5-second analysis window is used for good frequency resolution
- Power estimates are timestamped at the center of their analysis window
- Line noise (50/60Hz) is removed using notch filtering
- Signal quality is monitored in real-time

---

## Development Plan

1. Sync with video of freethrows on tripod, video shooter and the basket
2. Add prompts in the terminal to start and stop recording each shot
3. Add prompts to enter success or failure
4. Start the video synced with EEG
5. Record all EEG spectra not just alpha
6. Consider using a webpage interface for mobile access

## Dependencies

- OpenCV for video recording
- BrainFlow for EEG data
- NumPy for data processing
- Matplotlib for visualization
