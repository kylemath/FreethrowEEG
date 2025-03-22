"""
Main application class for the FreethrowEEG application.
"""

import matplotlib.pyplot as plt
import threading
import collections
import queue
import time
import numpy as np
import os
import json
from datetime import datetime

from src.utils.logger import logger
from src.utils.config import (
    BUFFER_LENGTH, FREQ_BANDS, FRAME_WIDTH, FRAME_HEIGHT,
    PRE_SHOT_DURATION, SHOT_DURATION, POST_SHOT_DURATION
)
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
        
        # Session data storage with original format
        self.session_data = {
            'player_id': None,
            'timestamp': None,
            'shots': [],
            'video_path': None,
            'eeg_data': {
                'timestamps': [],
                'bands': {band: [] for band in FREQ_BANDS.keys()}
            }
        }
        
        # Shot timing data
        self.shot_start_time = None
        self.current_shot_data = {
            'pre_shot': {band: [] for band in FREQ_BANDS.keys()},
            'during_shot': {band: [] for band in FREQ_BANDS.keys()},
            'post_shot': {band: [] for band in FREQ_BANDS.keys()}
        }
        
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
            
            # Initialize session data
            self.session_data['player_id'] = player_id
            self.session_data['timestamp'] = datetime.now().isoformat()
            self.session_data['shots'] = []
            
            # Set up video path and start recording
            session_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', f"session_{player_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(session_dir, exist_ok=True)
            self.session_data['video_path'] = os.path.join(session_dir, "session_recording.mp4")
            
            # Start video recording
            if not self.camera.start_recording(self.session_data['video_path']):
                raise RuntimeError("Failed to start video recording")
            
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
                while not self.data_queue.empty():
                    # Get data from the queue
                    current_time, powers = self.data_queue.get()
                    
                    # Store the timestamp and power values in their respective buffers
                    self.time_buffer.append(current_time)
                    for band, power in powers.items():
                        self.eeg_buffer[band].append(power)
                    
                    # Store data for saving in original format
                    self.session_data['eeg_data']['timestamps'].append(current_time)
                    for band, power in powers.items():
                        self.session_data['eeg_data']['bands'][band].append(float(power))
                    
                    # If we're in a shot sequence, store the data in the appropriate phase buffer
                    if self.shot_manager and self.shot_manager.recording_shot:
                        current_phase = self.shot_manager.current_recording_phase
                        
                        # Store the data in the appropriate phase buffer
                        if current_phase in ['pre_shot', 'during_shot', 'post_shot']:
                            for band, power in powers.items():
                                self.current_shot_data[current_phase][band].append(float(power))
                    
                    data_count += 1
                    
                    # Log data points periodically for debugging
                    if data_count % 100 == 0 and self.shot_manager and self.shot_manager.recording_shot:
                        logger.debug(f"Current data points - Pre: {len(next(iter(self.current_shot_data['pre_shot'].values())))}, "
                                   f"During: {len(next(iter(self.current_shot_data['during_shot'].values())))}, "
                                   f"Post: {len(next(iter(self.current_shot_data['post_shot'].values())))}")
                
                # Small sleep to prevent CPU overload
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in plot update worker: {e}")
                time.sleep(0.1)  # Add delay on error to prevent spam
    
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
        """Handle shot phase changes."""
        logger.info(f"Shot phase changed to {phase} ({current_shot}/{total_shots})")
        
        # If starting a new shot recording, set the start time and reset data buffers
        if phase == "recording":
            self.shot_start_time = time.time()
            # Reset shot data buffers before starting new recording
            self.current_shot_data = {
                'pre_shot': {band: [] for band in FREQ_BANDS.keys()},
                'during_shot': {band: [] for band in FREQ_BANDS.keys()},
                'post_shot': {band: [] for band in FREQ_BANDS.keys()}
            }
        
        # Record shot result and save data
        if previous_result:
            logger.info(f"Saving shot {current_shot} data (Result: {previous_result})")
            shot_data = {
                "shot_id": current_shot,
                "timestamp": time.time(),
                "success": previous_result == "made",
                "eeg_data": {
                    "pre_shot": {band: list(data) for band, data in self.current_shot_data['pre_shot'].items()},
                    "during_shot": {band: list(data) for band, data in self.current_shot_data['during_shot'].items()},
                    "post_shot": {band: list(data) for band, data in self.current_shot_data['post_shot'].items()}
                },
                "video_timestamps": {
                    "start": self.shot_start_time,
                    "end": time.time()
                },
                "duration": time.time() - self.shot_start_time if self.shot_start_time else None,
                "video_path": self.session_data['video_path']
            }
            
            # Log data points collected for debugging
            logger.info(f"Data points collected - Pre: {len(next(iter(self.current_shot_data['pre_shot'].values())))}, "
                       f"During: {len(next(iter(self.current_shot_data['during_shot'].values())))}, "
                       f"Post: {len(next(iter(self.current_shot_data['post_shot'].values())))}")
            
            self.session_data['shots'].append(shot_data)
            self.shot_start_time = None
        
        # Update interface
        if self.main_interface:
            self.main_interface.update_shot_controls(phase, current_shot, total_shots)
        
        # Handle session completion
        if phase == "complete" or (current_shot >= total_shots and phase != "review"):
            logger.info("Session complete, saving data and cleaning up...")
            # Stop video recording before saving data
            self.camera.stop_recording()
            self.save_session_data()
            self.cleanup()
    
    def save_session_data(self):
        """Save session data to file."""
        try:
            # Create session directory if it doesn't exist
            session_dir = os.path.dirname(self.session_data['video_path'])
            os.makedirs(session_dir, exist_ok=True)
            
            # Save session data
            data_file = os.path.join(session_dir, "session_data.json")
            with open(data_file, 'w') as f:
                json.dump(self.session_data, f, indent=2)
            
            logger.info(f"Session data saved to {data_file}")
            
        except Exception as e:
            logger.error(f"Error saving session data: {e}", exc_info=True)
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up resources...")
        
        # Use a list to track any errors during cleanup
        cleanup_errors = []
        
        try:
            # First stop any ongoing processes
            if hasattr(self, 'shot_manager') and self.shot_manager is not None:
                try:
                    self.shot_manager.cancel_timers()
                except Exception as e:
                    cleanup_errors.append(f"Error canceling timers: {e}")
            
            # Stop video recording before closing interface (to ensure all frames are saved)
            if hasattr(self, 'camera') and self.camera is not None:
                try:
                    self.camera.stop_recording()
                except Exception as e:
                    cleanup_errors.append(f"Error stopping video recording: {e}")
            
            # Save any remaining data
            if self.setup_complete:
                try:
                    self.save_session_data()
                except Exception as e:
                    cleanup_errors.append(f"Error saving session data: {e}")
            
            # Close the interface before stopping data collection
            if hasattr(self, 'main_interface') and self.main_interface is not None:
                try:
                    self.main_interface.close()
                except Exception as e:
                    cleanup_errors.append(f"Error closing interface: {e}")
            
            # Stop data collection
            if hasattr(self, 'eeg_processor') and self.eeg_processor is not None:
                try:
                    self.eeg_processor.stop()
                except Exception as e:
                    cleanup_errors.append(f"Error stopping EEG processor: {e}")
            
            # Release camera last
            if hasattr(self, 'camera') and self.camera is not None:
                try:
                    self.camera.release()
                except Exception as e:
                    cleanup_errors.append(f"Error releasing camera: {e}")
            
            # Close all matplotlib figures
            try:
                plt.close('all')
            except Exception as e:
                cleanup_errors.append(f"Error closing matplotlib figures: {e}")
            
            # Log any errors that occurred during cleanup
            if cleanup_errors:
                logger.warning("Some non-critical errors occurred during cleanup:")
                for error in cleanup_errors:
                    logger.warning(error)
            else:
                logger.info("Cleanup completed successfully")
                
        except Exception as e:
            logger.error(f"Critical error during cleanup: {e}", exc_info=True)
            # Try one last time to close all figures
            try:
                plt.close('all')
            except:
                pass 