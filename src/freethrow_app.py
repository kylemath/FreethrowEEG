"""
Main application class for the FreethrowEEG application.
"""

import matplotlib.pyplot as plt
import threading
import collections
import queue
import time
import numpy as np

from src.utils.logger import logger
from src.utils.config import BUFFER_LENGTH, FREQ_BANDS, FRAME_WIDTH, FRAME_HEIGHT
from src.eeg.processor import EEGProcessor
from src.video.camera import CameraManager
from src.ui.setup_dialog import SetupDialog
from src.ui.main_interface import MainInterface
from src.ui.visualization import EEGVisualizer, VideoVisualizer
from src.session.shot_manager import ShotManager

class FreethrowApp:
    """Main application class for FreethrowEEG."""
    
    def __init__(self):
        """Initialize the application."""
        self.setup_complete = False
        self.player_id = None
        self.num_shots = None
        
        # Initialize data buffers
        self.data_queue = queue.Queue()
        self.eeg_buffer = {band: collections.deque(maxlen=BUFFER_LENGTH) 
                          for band in FREQ_BANDS.keys()}
        self.time_buffer = collections.deque(maxlen=BUFFER_LENGTH)
        
        # Initialize components
        self.eeg_processor = EEGProcessor(self.data_queue)
        self.camera = CameraManager()
        
        # These will be initialized later
        self.setup_dialog = None
        self.main_interface = None
        self.eeg_visualizer = None
        self.video_visualizer = None
        self.shot_manager = None
        self.update_thread = None
        self.artists = []
    
    def start(self):
        """Start the application."""
        logger.info("Starting FreethrowEEG application")
        
        # Initialize setup dialog
        self.setup_dialog = SetupDialog(
            on_connect_muse=self.connect_muse,
            on_connect_camera=self.connect_camera,
            on_start_session=self.start_session,
            on_check_ready=self.check_ready
        )
        
        # Show setup dialog
        self.setup_dialog.show()
    
    def connect_muse(self, debug=False):
        """
        Connect to MUSE device or enable debug mode.
        
        Args:
            debug (bool): If True, use simulated data instead of real device.
            
        Returns:
            bool: True if connection successful, False otherwise.
        """
        return self.eeg_processor.connect_muse(debug)
    
    def connect_camera(self):
        """
        Connect to camera.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        return self.camera.connect()
    
    def check_ready(self):
        """
        Check if all setup is complete.
        """
        try:
            player_ready = len(self.setup_dialog.get_player_id()) > 0
            shots_ready = self.setup_dialog.get_num_shots() > 0
            
            muse_ready = self.eeg_processor.is_connected()
            camera_ready = self.camera.is_connected()
            
            logger.info(f"Setup status - Player: {player_ready}, Shots: {shots_ready}, MUSE: {muse_ready}, Camera: {camera_ready}")
            
            self.setup_dialog.update_ready_status(
                player_ready, shots_ready, muse_ready, camera_ready
            )
            
        except Exception as e:
            logger.error(f"Error in check_ready: {e}", exc_info=True)
    
    def start_session(self, player_id, num_shots):
        """
        Start the main session.
        
        Args:
            player_id (str): Player identifier.
            num_shots (int): Number of shots to record.
        """
        try:
            logger.info("Starting session...")
            self.player_id = player_id
            self.num_shots = num_shots
            self.setup_complete = True
            
            logger.info(f"Session info - Player: {self.player_id}, Shots: {self.num_shots}")
            
            # Initialize visualizers
            self.eeg_visualizer = EEGVisualizer(self.time_buffer, self.eeg_buffer)
            
            # Use the configured frame dimensions instead of querying the camera
            # This avoids potential float conversion issues
            self.video_visualizer = VideoVisualizer(FRAME_HEIGHT, FRAME_WIDTH)
            
            # Initialize shot manager
            self.shot_manager = ShotManager(
                on_shot_phase_change=self.handle_shot_phase_change,
                num_shots=self.num_shots
            )
            
            # Initialize main interface
            self.main_interface = MainInterface(
                on_start_shot=self.shot_manager.start_shot,
                on_mark_shot=self.shot_manager.mark_shot,
                on_animate=self.animate
            )
            
            # Start data collection
            self.start_data_collection()
            
            # Show main interface
            self.main_interface.show()
            
        except Exception as e:
            logger.error(f"Error starting session: {e}", exc_info=True)
            self.setup_complete = False
            self.cleanup()
    
    def start_data_collection(self):
        """Start data collection and plot update threads."""
        logger.info("Starting data collection...")
        
        # Start EEG data collection
        self.eeg_processor.start_data_collection()
        
        # Start plot update thread
        self.update_thread = threading.Thread(target=self.update_plots, daemon=True)
        self.update_thread.start()
        logger.info("Plot update thread started")
        
        # Wait a moment for initial data
        logger.info("Waiting for initial data...")
        time.sleep(0.5)
    
    def update_plots(self):
        """Update plots continuously."""
        logger.info("Starting plot update worker...")
        data_count = 0
        while True:
            try:
                # Process any new data
                items_processed = 0
                while not self.data_queue.empty():
                    # Get data from the queue - now the timestamp is the center of the analysis window
                    current_time, powers = self.data_queue.get()
                    
                    # Store the timestamp and power values in their respective buffers
                    self.time_buffer.append(current_time)
                    for band, power in powers.items():
                        self.eeg_buffer[band].append(power)
                    
                    items_processed += 1
                    data_count += 1
                
                if items_processed > 0:
                    logger.info(f"Processed {items_processed} new data points (total: {data_count})")
                    logger.info(f"Buffer sizes: time={len(self.time_buffer)}, delta={len(self.eeg_buffer['delta'])}")
                
                # Small sleep to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in plot update loop: {e}", exc_info=True)
                time.sleep(0.1)
    
    def animate(self, frame):
        """
        Animation function for matplotlib.
        
        Args:
            frame: Animation frame number.
            
        Returns:
            list: Artists to redraw.
        """
        try:
            logger.debug(f"Animate frame {frame}")
            
            updated_artists = []
            
            # First time setup
            if not self.artists:
                # Get axes from the main interface
                eeg_ax, quality_ax, video_ax = self.main_interface.get_axes()
                
                # Setup plots and collect artists
                updated_artists.extend(self.eeg_visualizer.setup_eeg_plot(eeg_ax))
                updated_artists.extend(self.eeg_visualizer.setup_quality_plot(quality_ax))
                updated_artists.extend(self.video_visualizer.setup_video_plot(video_ax))
                
                self.artists = updated_artists
                logger.info(f"Initial setup with {len(self.artists)} artists")
                return self.artists
            
            # Update EEG plot
            updated_artists.extend(self.eeg_visualizer.update_eeg_plot())
            
            # Update signal quality
            if self.eeg_processor.debug_mode:
                # In debug mode, show simulated good signal quality
                updated_artists.extend(self.eeg_visualizer.update_simulated_quality())
            else:
                try:
                    eeg_data = self.eeg_processor.get_signal_quality()
                    if eeg_data is not None and eeg_data.size > 0:
                        updated_artists.extend(
                            self.eeg_visualizer.update_signal_quality(
                                eeg_data, self.eeg_processor.eeg_channels
                            )
                        )
                except Exception as e:
                    logger.error(f"Error updating signal quality: {e}")
            
            # Update video frame
            try:
                frame = self.camera.get_frame()
                if frame is not None:
                    updated_artists.extend(self.video_visualizer.update_video_frame(frame))
                else:
                    logger.warning("Failed to read from camera")
            except Exception as e:
                logger.error(f"Error updating video: {e}")
            
            return updated_artists
            
        except Exception as e:
            logger.error(f"Error in animation update: {e}", exc_info=True)
            return []
    
    def handle_shot_phase_change(self, phase, current_shot, total_shots, previous_result=None):
        """
        Handle shot phase changes.
        
        Args:
            phase (str): New shot phase.
            current_shot (int): Current shot number.
            total_shots (int): Total number of shots.
            previous_result (str, optional): Result of the previous shot.
        """
        logger.info(f"Shot phase changed to {phase} ({current_shot}/{total_shots})")
        if self.main_interface:
            self.main_interface.update_shot_controls(phase, current_shot, total_shots)
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        try:
            # Cancel any active timers
            if hasattr(self, 'shot_manager') and self.shot_manager is not None:
                self.shot_manager.cancel_timers()
            
            # Stop EEG processor
            if hasattr(self, 'eeg_processor') and self.eeg_processor is not None:
                self.eeg_processor.stop()
            
            # Release camera
            if hasattr(self, 'camera') and self.camera is not None:
                self.camera.release()
            
            # Close main interface
            if hasattr(self, 'main_interface') and self.main_interface is not None:
                self.main_interface.close()
            
            # Close all matplotlib figures
            logger.debug("Closing all figures")
            plt.close('all')
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)
            # Try to force matplotlib to close all figures
            try:
                plt.close('all')
            except:
                pass 