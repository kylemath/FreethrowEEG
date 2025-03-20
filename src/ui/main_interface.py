"""
Main interface for the FreethrowEEG application.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import threading

from src.utils.config import FRAME_WIDTH, FRAME_HEIGHT
from src.utils.logger import logger

class MainInterface:
    """Main interface for the FreethrowEEG application."""
    
    def __init__(self, on_start_shot, on_mark_shot, on_animate):
        """
        Initialize the main interface.
        
        Args:
            on_start_shot (callable): Callback for starting a shot.
            on_mark_shot (callable): Callback for marking a shot as made or missed.
            on_animate (callable): Callback for animation updates.
        """
        self.on_start_shot = on_start_shot
        self.on_mark_shot = on_mark_shot
        self.on_animate = on_animate
        
        self.fig = None
        self.eeg_ax = None
        self.quality_ax = None
        self.video_ax = None
        
        self.start_shot_button = None
        self.success_button = None
        self.fail_button = None
        self.status_text = None
        
        self.animation = None
    
    def show(self):
        """Show the main interface."""
        try:
            logger.info("Creating main interface...")
            
            # Create figure
            self.fig = plt.figure(figsize=(15, 8))
            self.fig.canvas.manager.set_window_title('FreethrowEEG Session')
            
            # Create subplots
            self.eeg_ax = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
            self.quality_ax = plt.subplot2grid((3, 3), (0, 2), rowspan=1)
            self.video_ax = plt.subplot2grid((3, 3), (1, 2), rowspan=1)
            
            # Add control buttons
            self.start_shot_button = Button(plt.axes([0.7, 0.15, 0.2, 0.08]), 'Start Shot')
            self.start_shot_button.on_clicked(self._start_shot_handler)
            
            self.success_button = Button(plt.axes([0.7, 0.05, 0.1, 0.08]), 'Made ✓')
            self.success_button.on_clicked(lambda x: self._mark_shot_handler(x, True))
            self.success_button.set_active(False)
            
            self.fail_button = Button(plt.axes([0.8, 0.05, 0.1, 0.08]), 'Miss ✗')
            self.fail_button.on_clicked(lambda x: self._mark_shot_handler(x, False))
            self.fail_button.set_active(False)
            
            # Add status text
            self.status_text = plt.figtext(0.1, 0.05, 'Ready to start', size=10)
            
            # Create animation
            logger.info("Creating animation...")
            self.animation = FuncAnimation(
                self.fig, 
                self.on_animate, 
                interval=100, 
                blit=True, 
                save_count=100
            )
            
            # Show the figure
            logger.info("Showing main interface")
            plt.show()
            
        except Exception as e:
            logger.error(f"Error showing main interface: {e}", exc_info=True)
            # Try to clean up if there's an error
            if hasattr(self, 'fig'):
                plt.close(self.fig)
            raise
    
    def _start_shot_handler(self, event):
        """Handle start shot button events."""
        logger.info("Start shot button clicked")
        self.on_start_shot()
    
    def _mark_shot_handler(self, event, success):
        """Handle shot marking button events."""
        logger.info(f"Shot marked as {'successful' if success else 'missed'}")
        self.on_mark_shot(success)
    
    def update_status(self, text):
        """
        Update the status text.
        
        Args:
            text (str): New status text.
        """
        self.status_text.set_text(text)
    
    def update_shot_controls(self, phase, current_shot, total_shots):
        """
        Update the shot control buttons based on the current phase.
        
        Args:
            phase (str): Current shot phase.
            current_shot (int): Current shot number.
            total_shots (int): Total number of shots.
        """
        if phase == "prep":
            self.start_shot_button.set_active(False)
            self.success_button.set_active(False)
            self.fail_button.set_active(False)
            self.update_status('Get ready...')
        
        elif phase == "recording":
            self.start_shot_button.set_active(False)
            self.success_button.set_active(False)
            self.fail_button.set_active(False)
            self.update_status('SHOOT NOW!')
        
        elif phase == "review":
            self.start_shot_button.set_active(False)
            self.success_button.set_active(True)
            self.fail_button.set_active(True)
            self.update_status('Was it good?')
        
        elif phase == "ready":
            if current_shot >= total_shots:
                self.start_shot_button.set_active(False)
                self.success_button.set_active(False)
                self.fail_button.set_active(False)
                self.update_status('All shots completed!')
            else:
                self.start_shot_button.set_active(True)
                self.success_button.set_active(False)
                self.fail_button.set_active(False)
                result_text = "" if current_shot == 0 else "Shot MADE - " if phase == "made" else "Shot MISSED - "
                self.update_status(f'{result_text}Press Start Shot when ready')
    
    def get_axes(self):
        """
        Get all the axes objects.
        
        Returns:
            tuple: (eeg_ax, quality_ax, video_ax)
        """
        return self.eeg_ax, self.quality_ax, self.video_ax
    
    def close(self):
        """Close the interface."""
        if self.animation is not None:
            self.animation.event_source.stop()
        
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig) 