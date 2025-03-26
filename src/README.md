# FreethrowEEG Architecture

This directory contains the refactored FreethrowEEG application code, organized into a modular structure for better maintainability and extensibility.

## Directory Structure

```
src/
├── __init__.py            # Main package initialization
├── main.py                # Main entry point
├── freethrow_app.py       # Core application class
├── audio/                 # Audio generation and playback
├── eeg/                   # EEG data processing
├── session/               # Shot and session management
├── ui/                    # User interface components
├── utils/                 # Utilities and configuration
└── video/                 # Camera and video processing
```

## Component Overview

### Core Components

- **main.py**: Entry point that initializes and starts the application
- **freethrow_app.py**: Central controller class that coordinates all application components

### Utility Components

- **utils/config.py**: Configuration parameters for the application (frequency bands, timing settings, etc.)
- **utils/logger.py**: Logging configuration and setup

### Data Processing

- **eeg/processor.py**: Handles connection to MUSE headband, data collection, and frequency analysis
- **video/camera.py**: Manages camera connection and frame capture

### User Interface

- **ui/setup_dialog.py**: Initial setup dialog for player info, MUSE and camera connection
- **ui/main_interface.py**: Main application interface with shot controls
- **ui/visualization.py**: Visualization components for EEG data and video feed

### Session Management

- **session/shot_manager.py**: Manages the timing and sequence of shots
- **audio/beep_generator.py**: Generates audio cues for shot timing

## How It Works

1. **Initialization**: The application starts in `main.py`, creating a `FreethrowApp` instance
2. **Setup Phase**: The setup dialog appears, letting the user connect to MUSE and camera
3. **Session Phase**: After setup, the main interface appears with EEG visualization and shot controls
4. **Data Flow**:
   - EEG data is continuously collected by `EEGProcessor` and placed in a queue
   - A worker thread processes this data and updates data buffers
   - The visualization components read from these buffers and update the UI

## Customization

### Configuration Parameters

All configurable settings are centralized in `utils/config.py`, including:

- **EEG Settings**: Frequency bands, sampling rates, signal thresholds
- **Video Settings**: Frame dimensions and frame rate
- **Shot Timing**: Preparation time, shot duration, etc.
- **Audio Settings**: Sample rate, beep duration, etc.

### Adding New Features

- **New Visualizations**: Extend the visualization classes in `ui/visualization.py`
- **Additional EEG Processing**: Add methods to `eeg/processor.py`
- **Custom Shot Sequence**: Modify `session/shot_manager.py`

## Dependencies

- **brainflow**: For EEG data collection and processing
- **matplotlib**: For visualization and UI
- **numpy**: For data processing
- **OpenCV (cv2)**: For camera capture
- **sounddevice**: For audio playback

## Debugging

To run the application in debug mode (without MUSE hardware):

1. Start the application normally
2. In the setup dialog, click "Debug Mode" instead of "Connect MUSE"
3. Connect a camera (or the application will attempt to use webcam)
4. This will generate simulated EEG data for testing

## Logging

The application uses Python's standard logging module, configured in `utils/logger.py`. Log levels can be adjusted there to increase or decrease verbosity.
