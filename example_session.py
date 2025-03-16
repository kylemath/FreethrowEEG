from session_manager import Session
import time
import random

# Create a mock session
session = Session("ExamplePlayer", 2)

# Simulate two shots with mock EEG data
for shot_id in [1, 2]:
    # Simulate some EEG data
    mock_eeg_data = {
        'alpha': [random.random() for _ in range(10)],
        'beta': [random.random() for _ in range(10)],
        'theta': [random.random() for _ in range(10)],
        'delta': [random.random() for _ in range(10)],
        'gamma': [random.random() for _ in range(10)]
    }
    
    # Create shot data
    shot_data = {
        "shot_id": shot_id,
        "timestamp": time.time(),
        "success": shot_id == 1,  # First shot success, second failure
        "eeg_data": mock_eeg_data,
        "duration": 3.5  # Example duration in seconds
    }
    
    # Add shot to session
    session.add_shot(shot_data)
    print(f"Added shot {shot_id}")
    time.sleep(1)  # Small delay between shots

print(f"\nSession data saved in: {session.session_dir}") 