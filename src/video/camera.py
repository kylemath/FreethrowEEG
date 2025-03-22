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
                    
                    # Set camera properties
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                    self.cap.set(cv2.CAP_PROP_FPS, FPS)
                    
                    return True
                cap.release()
        
        logger.error("Could not connect to camera")
        return False
    
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
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                FPS, 
                (int(FRAME_WIDTH), int(FRAME_HEIGHT))
            )
            self.recording = True
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
        
        ret, frame = self.cap.read()
        if ret:
            # Write frame to video if recording
            if self.recording and self.video_writer is not None:
                self.video_writer.write(frame)
            return frame
        return None
    
    def stop_recording(self):
        """Stop video recording."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            self.recording = False
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