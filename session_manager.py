import json
import datetime
import os
from pathlib import Path
import shutil

class Session:
    def __init__(self, player_id, num_shots):
        self.player_id = player_id
        self.num_shots = num_shots
        self.date_time = datetime.datetime.now()
        self.shots = []
        
        # Create session directory
        self.session_dir = self._create_session_directory()
        
    def _create_session_directory(self):
        """Create a directory for storing session data"""
        try:
            # Create data directory if it doesn't exist
            base_dir = Path("data")
            base_dir.mkdir(exist_ok=True)
            
            # Create player directory
            player_dir = base_dir / self.player_id
            player_dir.mkdir(exist_ok=True)
            
            # Create session-specific directory with timestamp
            session_name = self.date_time.strftime('%Y%m%d_%H%M%S')
            session_dir = player_dir / session_name
            session_dir.mkdir(exist_ok=True)
            
            return session_dir
        except Exception as e:
            print(f"Error creating directory structure: {e}")
            raise
    
    def safe_save_json(self, data, filepath):
        """Safely save JSON data using a temporary file"""
        temp_file = filepath.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=4)
            shutil.move(str(temp_file), str(filepath))
        except Exception as e:
            print(f"Error saving data to {filepath}: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def save_metadata(self):
        """Save session metadata to JSON file"""
        metadata = {
            "session_info": {
                "player_id": self.player_id,
                "date_time": self.date_time.isoformat(),
                "total_shots": self.num_shots
            },
            "shots": self.shots  # Include full shot data in metadata
        }
        
        # Save metadata
        metadata_file = self.session_dir / "session_metadata.json"
        self.safe_save_json(metadata, metadata_file)
            
        # Also save shots separately for easier access
        shots_file = self.session_dir / "shots.json"
        self.safe_save_json({"shots": self.shots}, shots_file)
            
    def add_shot(self, shot_data):
        """Add shot data to the session"""
        try:
            # Ensure shot data has the required structure
            required_fields = {
                "shot_id": int,
                "timestamp": float,
                "success": bool,
                "eeg_data": {
                    "pre_shot": dict,
                    "during_shot": dict,
                    "post_shot": dict
                },
                "video_path": str
            }
            
            # Validate shot data structure
            for field, field_type in required_fields.items():
                if field not in shot_data:
                    raise ValueError(f"Missing required field: {field}")
                if field != "eeg_data" and not isinstance(shot_data[field], field_type):
                    raise ValueError(f"Invalid type for {field}. Expected {field_type}")
            
            # Validate EEG data structure
            for period in ["pre_shot", "during_shot", "post_shot"]:
                if period not in shot_data["eeg_data"]:
                    raise ValueError(f"Missing EEG data for period: {period}")
            
            self.shots.append(shot_data)
            self.save_metadata()  # Update metadata after each shot
        except Exception as e:
            print(f"Error adding shot data: {e}")
            raise

def initialize_session():
    """Interactive function to initialize a new session"""
    print("\n=== New FreethrowEEG Session ===\n")
    
    # Get player information
    player_id = input("Enter player name/ID: ").strip()
    while not player_id:
        print("Player name/ID cannot be empty!")
        player_id = input("Enter player name/ID: ").strip()
    
    # Get number of planned shots
    while True:
        try:
            num_shots = int(input("Enter number of planned shots: "))
            if num_shots > 0:
                break
            print("Number of shots must be positive!")
        except ValueError:
            print("Please enter a valid number!")
    
    # Create and return new session
    session = Session(player_id, num_shots)
    print(f"\nSession initialized for {player_id}")
    print(f"Session directory: {session.session_dir}")
    print(f"Planned shots: {num_shots}")
    print("\nSession is ready to begin!")
    
    return session 