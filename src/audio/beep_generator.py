"""
Audio beep generation and playback functionality.
"""

import numpy as np
import sounddevice as sd
from src.utils.config import SAMPLE_RATE, BEEP_DURATION
from src.utils.logger import logger

def generate_beep(frequency=1000, duration=None):
    """
    Generate a beep sound.
    
    Args:
        frequency (int): Frequency of the beep in Hz.
        duration (float): Duration of the beep in seconds. If None, uses default BEEP_DURATION.
        
    Returns:
        numpy.ndarray: Audio data for the beep.
    """
    if duration is None:
        duration = BEEP_DURATION
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    return np.sin(2 * np.pi * frequency * t)

def play_beep_nonblocking(beep):
    """
    Play a beep sound without blocking.
    
    Args:
        beep (numpy.ndarray): Audio data to play.
    """
    try:
        sd.play(beep, SAMPLE_RATE, blocking=False)
    except Exception as e:
        logger.error(f"Error playing beep: {e}")

class BeepManager:
    """Manager for creating and playing audio cues."""
    
    def __init__(self):
        """Initialize audio cues."""
        self.ready_beep = generate_beep(frequency=800)
        self.shot_beep = generate_beep(frequency=1200)
    
    def play_ready_beep(self):
        """Play the ready beep sound."""
        play_beep_nonblocking(self.ready_beep)
    
    def play_shot_beep(self):
        """Play the shot beep sound."""
        play_beep_nonblocking(self.shot_beep) 