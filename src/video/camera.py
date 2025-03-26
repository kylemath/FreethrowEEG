"""
Camera management functionality.
"""

import cv2
import time
from src.utils.logger import logger
from src.utils.config import FRAME_WIDTH, FRAME_HEIGHT, FPS

class CameraManager:
    """Manages camera connection and frame capture."""
    
    def __init__(self):
        """Initialize the camera manager."""
        self.cap = None
        self.video_writer = None
        self.recording = False
        self.last_frame_time = None
        self.frame_count = 0
        self.start_time = None
        self.target_frame_interval = 1.0 / FPS  # Time between frames in seconds
    
    def connect(self):
        """
        Connect to camera.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        logger.info("Attempting to connect to camera...")
        
        # Try different camera indices
        for camera_index in [1, 0, 2]:
            logger.info(f"Trying camera index {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            
            # Wait for camera to initialize
            time.sleep(1)
            
            if cap.isOpened():
                # Try to read a test frame
                ret, frame = cap.read()
                if ret:
                    logger.info(f"Camera connected on index {camera_index}")
                    self.cap = cap
                    
                    # Set camera properties with verification
                    self._set_camera_property(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    self._set_camera_property(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    self._set_camera_property(cv2.CAP_PROP_FPS, FPS)
                    
                    # Set camera buffer size to 1 to get the most recent frame
                    self._set_camera_property(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Verify settings
                    actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    logger.info(f"Camera configured - FPS: {actual_fps}, "
                              f"Resolution: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
                    
                    return True
                cap.release()
        
        logger.error("Could not connect to camera")
        return False
    
    def _set_camera_property(self, prop_id, value):
        """Set camera property and verify it was set correctly."""
        self.cap.set(prop_id, value)
        actual_value = self.cap.get(prop_id)
        logger.info(f"Camera property {prop_id} - Requested: {value}, Actual: {actual_value}")
    
    def start_recording(self, output_path):
        """
        Start recording video.
        
        Args:
            output_path (str): Path to save the video file.
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Cannot start recording - camera not connected")
            return False
        
        try:
            # Use H.264 codec for better compression and compatibility
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                FPS, 
                (int(FRAME_WIDTH), int(FRAME_HEIGHT))
            )
            
            if not self.video_writer.isOpened():
                # Fallback to MP4V codec if H.264 is not available
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(
                    output_path, 
                    fourcc, 
                    FPS, 
                    (int(FRAME_WIDTH), int(FRAME_HEIGHT))
                )
            
            self.recording = True
            self.frame_count = 0
            self.start_time = time.time()
            self.last_frame_time = self.start_time
            logger.info(f"Started video recording to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting video recording: {e}")
            return False
    
    def get_frame(self):
        """
        Get the current frame from the camera.
        
        Returns:
            numpy.ndarray: Current frame, or None if capture failed.
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        current_time = time.time()
        
        # If we're recording, ensure proper frame timing
        if self.recording and self.last_frame_time is not None:
            time_since_last_frame = current_time - self.last_frame_time
            if time_since_last_frame < self.target_frame_interval:
                # Wait until it's time for the next frame
                time.sleep(self.target_frame_interval - time_since_last_frame)
                current_time = time.time()
        
        # Read frame
        ret, frame = self.cap.read()
        
        if ret:
            # Write frame to video if recording
            if self.recording and self.video_writer is not None:
                self.video_writer.write(frame)
                self.frame_count += 1
                self.last_frame_time = current_time
                
                # Log frame rate every 5 seconds
                elapsed_time = current_time - self.start_time
                if elapsed_time >= 5 and self.frame_count % (FPS * 5) == 0:
                    actual_fps = self.frame_count / elapsed_time
                    logger.info(f"Recording stats - Frames: {self.frame_count}, "
                              f"Time: {elapsed_time:.1f}s, FPS: {actual_fps:.1f}")
            
            return frame
        return None
    
    def stop_recording(self):
        """Stop video recording."""
        if self.video_writer is not None:
            # Log final recording stats
            if self.start_time is not None:
                elapsed_time = time.time() - self.start_time
                actual_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                logger.info(f"Recording complete - Total frames: {self.frame_count}, "
                          f"Duration: {elapsed_time:.1f}s, Average FPS: {actual_fps:.1f}")
            
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
            self.frame_count = 0
            self.start_time = None
            self.last_frame_time = None
            logger.info("Stopped video recording")
    
    def is_connected(self):
        """
        Check if camera is connected.
        
        Returns:
            bool: True if camera is connected, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()
    
    def release(self):
        """Release camera resources."""
        logger.info("Releasing camera resources...")
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None 