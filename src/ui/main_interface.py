"""
Main interface for the FreethrowEEG application.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.animation import FuncAnimation
import threading

from src.utils.config import FRAME_WIDTH, FRAME_HEIGHT
from src.utils.logger import logger

class CustomButton(Button):
    """Custom button class with enhanced visual feedback."""
    
    def __init__(self, ax, label, color='lightgray'):
        """Initialize custom button with enhanced styling."""
        super().__init__(ax, label, color=color, hovercolor='0.9')
        self.label.set_fontsize(10)
        self.label.set_weight('bold')
        self._active_color = color
        self._inactive_color = '0.85'  # Light gray
        self._pressed_color = '0.7'    # Darker gray for pressed state
        self._is_pressed = False
        self._is_active = True
        
    def set_active(self, active):
        """Set button active/inactive state with visual feedback."""
        self._is_active = active
        if active:
            self.color = self._active_color
            self.hovercolor = '0.9'
            self.label.set_alpha(1.0)
        else:
            self.color = self._inactive_color
            self.hovercolor = self._inactive_color
            self.label.set_alpha(0.5)
        if self.ax:
            self.ax.set_facecolor(self.color)
            
    def _pressed(self, event):
        """Handle button press with visual feedback."""
        if self._is_active and event.inaxes == self.ax:
            self._is_pressed = True
            self.ax.set_facecolor(self._pressed_color)
            self.ax.figure.canvas.draw_idle()
            
    def _released(self, event):
        """Handle button release with visual feedback."""
        if self._is_pressed:
            self._is_pressed = False
            if self._is_active:
                if event.inaxes == self.ax:
                    self.ax.set_facecolor(self.hovercolor)
                else:
                    self.ax.set_facecolor(self.color)
                self.ax.figure.canvas.draw_idle()
                if event.inaxes == self.ax:
                    self._clicked(event)

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
            
            # Add control buttons with custom styling
            start_button_ax = plt.axes([0.7, 0.15, 0.2, 0.08])
            self.start_shot_button = CustomButton(start_button_ax, 'Start Shot', color='lightblue')
            self.start_shot_button.on_clicked(self._start_shot_handler)
            
            # Swap the positions of success and fail buttons
            fail_button_ax = plt.axes([0.7, 0.05, 0.1, 0.08])
            self.fail_button = CustomButton(fail_button_ax, 'Miss ✗', color='#ffb3b3')  # Light red
            self.fail_button.on_clicked(lambda x: self._mark_shot_handler(x, False))
            self.fail_button.set_active(False)
            
            success_button_ax = plt.axes([0.8, 0.05, 0.1, 0.08])
            self.success_button = CustomButton(success_button_ax, 'Made ✓', color='lightgreen')
            self.success_button.on_clicked(lambda x: self._mark_shot_handler(x, True))
            self.success_button.set_active(False)
            
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
                # Handle previous shot result text
                if current_shot == 0:
                    result_text = ""
                else:
                    # Use the explicit phase value for determining the result text
                    result_text = "Shot MADE - " if phase == "made" else "Shot MISSED - " if phase == "missed" else ""
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