"""
Visualization components for the FreethrowEEG application.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

from src.utils.config import (
    FREQ_BANDS, MUSE_ELECTRODE_POSITIONS,
    SIGNAL_THRESHOLD, MIN_SIGNAL_QUALITY
)
from src.utils.logger import logger

class EEGVisualizer:
    """Handles visualization of EEG data and signal quality."""
    
    def __init__(self, time_buffer, eeg_buffer):
        """
        Initialize the EEG visualizer.
        
        Args:
            time_buffer (collections.deque): Buffer for timestamps.
            eeg_buffer (dict): Dictionary of buffers for each frequency band.
        """
        self.time_buffer = time_buffer
        self.eeg_buffer = eeg_buffer
        self.eeg_ax = None
        self.quality_ax = None
        self.elapsed_ax = None
        self.lines = {}
        self.colors = {'delta': 'blue', 'theta': 'green', 'alpha': 'red',
                     'beta': 'purple', 'gamma': 'orange'}
        self.quality_circles = {}
        self.quality_text = {}
        self.legend_created = False
    
    def setup_eeg_plot(self, ax):
        """
        Set up the EEG frequency band power plot.
        
        Args:
            ax (matplotlib.axes.Axes): Axes to create the plot on.
        """
        self.eeg_ax = ax
        
        # Create lines for each frequency band
        for band in FREQ_BANDS.keys():
            line, = self.eeg_ax.plot([], [], label=band, color=self.colors[band])
            self.lines[band] = line
            logger.debug(f"Created line for {band}")
        
        # Set up log scale for power values
        self.eeg_ax.set_yscale('log')
        
        # Configure tick marks to be more readable
        self.eeg_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1e}'))
        
        # Create a twin x-axis for elapsed time
        self.elapsed_ax = self.eeg_ax.twinx()
        self.elapsed_ax.spines['right'].set_position(('outward', 60))  # Move it outward for clarity
        self.elapsed_ax.yaxis.set_visible(False)  # Hide y-axis of twin
        
        self.eeg_ax.set_xlabel('Time (s)')
        self.eeg_ax.set_ylabel('Power (log scale)')
        self.eeg_ax.set_title('EEG Frequency Band Powers')
        
        # Only create the legend once
        if not self.legend_created:
            self.eeg_ax.legend()
            self.legend_created = True
            
        self.eeg_ax.grid(True)
        self.eeg_ax.set_xlim(0, 10)
        self.eeg_ax.set_ylim(0.1, 1000)  # Set reasonable log scale limits
        
        return [self.eeg_ax, self.elapsed_ax]
    
    def setup_quality_plot(self, ax):
        """
        Set up the signal quality visualization.
        
        Args:
            ax (matplotlib.axes.Axes): Axes to create the plot on.
        """
        self.quality_ax = ax
        self.quality_ax.clear()
        
        # Draw head outline
        circle = plt.Circle((0, 0.3), 0.8, fill=False, color='gray')
        self.quality_ax.add_patch(circle)
        # Draw nose
        self.quality_ax.plot([0, 0], [1.1, 0.9], 'gray')
        
        # Create circles for each electrode
        self.quality_circles = {}
        self.quality_text = {}
        for name, pos in MUSE_ELECTRODE_POSITIONS.items():
            circle = plt.Circle(pos, 0.1, color='gray', alpha=0.5)
            text = self.quality_ax.text(pos[0], pos[1]-0.2, name, ha='center', va='center')
            self.quality_ax.add_patch(circle)
            self.quality_circles[name] = circle
            self.quality_text[name] = text
        
        self.quality_ax.set_xlim(-1.2, 1.2)
        self.quality_ax.set_ylim(-0.5, 1.5)
        self.quality_ax.axis('off')
        self.quality_ax.set_title('Signal Quality')
        
        return list(self.quality_circles.values())
    
    def update_eeg_plot(self):
        """
        Update the EEG frequency band power plot.
        
        Returns:
            list: Updated plot elements.
        """
        updated_elements = []
        
        # Update each frequency band line
        for band in FREQ_BANDS.keys():
            times = list(self.time_buffer)
            powers = list(self.eeg_buffer[band])
            
            if len(times) < 1:
                logger.debug(f"No data points for {band}: empty buffer")
                continue
                
            if len(times) > 1:  # Need at least 2 points to interpolate
                # Ensure times are strictly increasing
                try:
                    # Log before unique to help debug
                    logger.debug(f"Time buffer before unique: min={min(times):.2f}, max={max(times):.2f}, len={len(times)}")
                    times, unique_idx = np.unique(times, return_index=True)
                    if len(unique_idx) != len(powers):
                        logger.debug(f"Unique indices: {len(unique_idx)}, Powers: {len(powers)}")
                    
                    powers = np.array(powers)[unique_idx]
                    logger.debug(f"After unique: times={len(times)}, powers={len(powers)}")
                    
                    if len(times) > 0 and len(powers) > 0:
                        self.lines[band].set_data(times, powers)
                        updated_elements.append(self.lines[band])
                        logger.debug(f"Updated {band} line with {len(times)} points")
                    else:
                        logger.warning(f"Empty data after unique operation: times={len(times)}, powers={len(powers)}")
                except Exception as e:
                    logger.error(f"Error processing unique times for {band}: {e}")
                    # As a fallback, try using the data directly
                    if len(times) == len(powers):
                        self.lines[band].set_data(times, powers)
                        updated_elements.append(self.lines[band])
                        logger.debug(f"Fallback: Updated {band} line with {len(times)} direct points")
            else:
                logger.debug(f"Not enough data points for {band}: {len(times)}")
        
        # Adjust plot limits if we have data
        if len(self.time_buffer) > 0:
            try:
                # Get current time
                current_time = list(self.time_buffer)[-1] if self.time_buffer else 0
                logger.debug(f"Current plot time: {current_time:.2f}s")
                
                # Always show from 0 to current time with a small padding
                padding = max(5, current_time * 0.05)  # At least 5 seconds or 5% of total time
                xlim = (0, current_time + padding)
                self.eeg_ax.set_xlim(xlim)
                self.elapsed_ax.set_xlim(xlim)
                updated_elements.extend([self.eeg_ax, self.elapsed_ax])
                logger.debug(f"Set x limits to {xlim}")
                
                # Generate appropriate tick marks - distribute evenly
                if current_time > 0:
                    # Calculate appropriate tick interval based on the time span
                    if current_time < 30:
                        tick_interval = 5  # 5-second intervals for shorter recordings
                    elif current_time < 120:
                        tick_interval = 15  # 15-second intervals for 30s-2min recordings
                    elif current_time < 300:
                        tick_interval = 30  # 30-second intervals for 2-5min recordings
                    else:
                        tick_interval = 60  # 1-minute intervals for longer recordings
                    
                    # Create evenly spaced ticks
                    max_ticks = 10  # Limit number of ticks for readability
                    ticks = np.linspace(0, current_time, min(max_ticks, int(current_time/tick_interval) + 1))
                    
                    # Update main axis (left)
                    self.eeg_ax.set_xticks(ticks)
                    if current_time > 60:  # More than 1 minute
                        self.eeg_ax.set_xticklabels([f"{int(t/60)}:{int(t%60):02d}" for t in ticks])
                        self.eeg_ax.set_xlabel('Time (min:sec)')
                    else:
                        self.eeg_ax.set_xticklabels([f"{int(t)}" for t in ticks])
                        self.eeg_ax.set_xlabel('Time (sec)')
                    
                    # Update elapsed time axis (right)
                    self.elapsed_ax.set_xticks(ticks)
                    self.elapsed_ax.set_xticklabels([f"{int(t)}" for t in ticks])
                    self.elapsed_ax.set_xlabel('Elapsed Time (sec)', labelpad=10)
                    
                    logger.debug(f"Updated tick marks with {len(ticks)} ticks")
                    
                    # Force update of tick labels
                    # self.eeg_ax.figure.canvas.draw()
                    # self.elapsed_ax.figure.canvas.draw()
            except Exception as e:
                logger.error(f"Error adjusting plot limits: {e}")
        else:
            logger.debug("No data in time buffer for plot limits")
            
        return updated_elements
    
    def update_signal_quality(self, data, eeg_channels):
        """
        Update signal quality visualization.
        
        Args:
            data (numpy.ndarray): EEG data array.
            eeg_channels (list): List of EEG channel indices.
            
        Returns:
            list: Updated quality circles.
        """
        updated_elements = []
        
        for i, channel in enumerate(eeg_channels):
            if i >= len(MUSE_ELECTRODE_POSITIONS):
                break
            
            electrode = list(MUSE_ELECTRODE_POSITIONS.keys())[i]
            channel_data = data[channel]
            
            if len(channel_data) > 0:
                has_nans = np.isnan(channel_data).any()
                above_threshold = np.abs(channel_data) > SIGNAL_THRESHOLD
                quality = 0 if has_nans else np.sum(~above_threshold) / len(channel_data)
            else:
                quality = 0
            
            if quality > 0.8:
                color, alpha = 'green', 0.8
            elif quality > 0.5:
                color, alpha = 'yellow', 0.6
            else:
                color, alpha = 'red', 0.4
            
            circle = self.quality_circles[electrode]
            circle.set_color(color)
            circle.set_alpha(alpha)
            # Make sure the circle is visible
            if not circle.get_visible():
                circle.set_visible(True)
            updated_elements.append(circle)
        
        return updated_elements
    
    def update_simulated_quality(self):
        """
        Update signal quality visualization with simulated good quality.
        
        Returns:
            list: Updated quality circles.
        """
        updated_elements = []
        
        for electrode in MUSE_ELECTRODE_POSITIONS.keys():
            circle = self.quality_circles[electrode]
            circle.set_color('green')
            circle.set_alpha(0.8)
            if not circle.get_visible():
                circle.set_visible(True)
            updated_elements.append(circle)
            
        logger.debug("Updated simulated signal quality")
        return updated_elements


class VideoVisualizer:
    """Handles visualization of video feed."""
    
    def __init__(self, frame_height, frame_width):
        """
        Initialize the video visualizer.
        
        Args:
            frame_height (int): Height of video frames.
            frame_width (int): Width of video frames.
        """
        # Ensure dimensions are integers
        self.frame_height = int(frame_height)
        self.frame_width = int(frame_width)
        self.video_ax = None
        self.video_image = None
    
    def setup_video_plot(self, ax):
        """
        Set up the video feed visualization.
        
        Args:
            ax (matplotlib.axes.Axes): Axes to create the plot on.
        """
        self.video_ax = ax
        self.video_ax.set_title('Webcam Feed')
        self.video_ax.axis('off')
        
        # Initialize video image with black frame
        # Ensure dimensions are integers
        self.video_image = self.video_ax.imshow(
            np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        )
        return [self.video_image]
    
    def update_video_frame(self, frame):
        """
        Update the video feed with a new frame.
        
        Args:
            frame (numpy.ndarray): New video frame in BGR format.
            
        Returns:
            list: Updated video image.
        """
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.video_image.set_array(frame_rgb)
            logger.debug("Updated video frame")
            return [self.video_image]
        return [] 