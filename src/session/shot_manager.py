"""
Shot timing and recording management functionality.
"""

import time
import threading
from src.utils.config import PREP_TIME, PRE_SHOT_DURATION, SHOT_DURATION, POST_SHOT_DURATION
from src.utils.logger import logger
from src.audio.beep_generator import BeepManager

class ShotManager:
    """Manages the timing and recording of shots."""
    
    def __init__(self, on_shot_phase_change, num_shots):
        """
        Initialize the shot manager.
        
        Args:
            on_shot_phase_change (callable): Callback for shot phase changes.
            num_shots (int): Total number of shots to record.
        """
        self.on_shot_phase_change = on_shot_phase_change
        self.num_shots = num_shots
        
        self.current_shot = 0
        self.recording_shot = False
        self.shot_phase = "ready"
        self.phase_start_time = None
        self.shot_start_time = None
        self.shot_timer = None
        
        # Audio cues
        self.beep_manager = BeepManager()
    
    def start_shot(self):
        """Start the shot sequence."""
        if self.current_shot >= self.num_shots:
            logger.info("All shots completed!")
            self._update_phase("complete")
            return
        
        self._update_phase("prep")
        self.phase_start_time = time.time()
        
        # Start shot sequence timer
        self.shot_timer = threading.Timer(PREP_TIME, self.start_recording)
        self.shot_timer.start()
        
        # Schedule ready beep
        threading.Timer(PREP_TIME - 2, 
                      lambda: self.beep_manager.play_ready_beep()).start()
        
        logger.info(f"Starting shot {self.current_shot + 1} preparation")
    
    def start_recording(self):
        """Start recording the shot."""
        if self.shot_phase != "prep":
            return
        
        self._update_phase("recording")
        self.recording_shot = True
        self.shot_start_time = time.time()
        self.beep_manager.play_shot_beep()
        
        # Schedule end of recording
        self.shot_timer = threading.Timer(SHOT_DURATION, self.end_recording)
        self.shot_timer.start()
        
        logger.info("Shot recording started")
    
    def end_recording(self):
        """End the shot recording."""
        if self.shot_phase != "recording":
            return
        
        self._update_phase("review")
        self.recording_shot = False
        
        logger.info("Shot recording ended, waiting for result")
    
    def mark_shot(self, success):
        """
        Mark the shot as made or missed.
        
        Args:
            success (bool): True if shot was successful, False otherwise.
        """
        if self.shot_phase != "review":
            return
        
        self.current_shot += 1
        result = "made" if success else "missed"
        
        if self.current_shot >= self.num_shots:
            self._update_phase("complete")
        else:
            self._update_phase("ready", previous_result=result)
        
        logger.info(f"Shot {self.current_shot} marked as {result}")
    
    def _update_phase(self, phase, previous_result=None):
        """
        Update the shot phase and notify observers.
        
        Args:
            phase (str): New shot phase.
            previous_result (str, optional): Result of the previous shot.
        """
        self.shot_phase = phase
        self.on_shot_phase_change(phase, self.current_shot, self.num_shots, previous_result)
    
    def cancel_timers(self):
        """Cancel any active timers."""
        if self.shot_timer is not None:
            self.shot_timer.cancel()
            self.shot_timer = None 