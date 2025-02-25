# FreethrowEEG Implementation Plan

## Overview

This document outlines the plan to modify the existing EEG recording system to create a comprehensive freethrow shooting experiment that correlates brain activity with shooting performance.

## Core Features to Implement

### 1. Session Management

- Add session initialization prompts
  - Player name/ID
  - Session date/time
  - Number of planned shots
- Create a structured data storage system
  - Save raw EEG data
  - Save shot outcomes
  - Save session metadata

### 2. Shot Recording System

- Implement a shot-by-shot recording system:
  - Pre-shot baseline period (e.g., 5 seconds)
  - Shot execution period
  - Post-shot period
- Add keyboard controls:
  - Spacebar to start/stop each shot recording
  - Keys for recording shot success/failure (e.g., 'S' for success, 'F' for failure)

### 3. EEG Data Enhancement

- Expand beyond just alpha waves:
  - Record full EEG spectrum (delta, theta, alpha, beta, gamma)
  - Calculate power ratios between different bands
  - Track temporal changes in brain activity
- Implement real-time quality checks:
  - Signal quality indicators
  - Electrode connection status
  - Motion artifact detection

### 4. Video Integration

- Add video recording capabilities:
  - Synchronize video timestamps with EEG data
  - Record from multiple angles (player and basket)
  - Store video files with appropriate naming convention
- Create video triggers tied to shot recording system

### 5. Data Storage and Analysis

- Create a structured data format:

```python
{
    "session_info": {
        "player_id": str,
        "date_time": timestamp,
        "total_shots": int
    },
    "shots": [
        {
            "shot_id": int,
            "timestamp": timestamp,
            "success": bool,
            "eeg_data": {
                "pre_shot": array,
                "during_shot": array,
                "post_shot": array
            },
            "video_path": str
        }
    ]
}
```

- Implement automatic data export:
  - CSV files for EEG data
  - JSON for session metadata
  - MP4 for video recordings

### 6. Performance Analysis Tools

- Create analysis scripts for:
  - Success rate tracking
  - EEG pattern correlation with success
  - Learning curve visualization
  - Brain state progression over session

## Implementation Phases

### Phase 1: Basic Shot Recording

1. Modify existing code to handle shot-by-shot recording
2. Implement basic user interface for shot management
3. Add simple data storage system

### Phase 2: Enhanced Data Collection

1. Add video recording integration
2. Expand EEG band analysis
3. Implement quality monitoring

### Phase 3: Analysis and Visualization

1. Create data processing pipeline
2. Implement basic analysis tools
3. Add visualization capabilities

## Technical Considerations

- Use threading for simultaneous video and EEG recording
- Implement proper synchronization mechanisms
- Ensure data backup and integrity
- Handle error cases gracefully
- Minimize system resource usage

## Next Steps

1. Begin with Phase 1 implementation
2. Test basic functionality with mock shots
3. Gather user feedback
4. Iterate and enhance based on real usage

## Dependencies to Add

- OpenCV for video recording
- Pandas for data management
- Matplotlib for visualization
- PyQt or Tkinter for GUI elements (if needed)
