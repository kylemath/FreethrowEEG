"""
Camera capture functionality.
"""

import cv2
from src.utils.config import FRAME_WIDTH, FRAME_HEIGHT, FPS
from src.utils.logger import logger

class CameraManager:
    """Handles camera capture operations."""
    
    def __init__(self):
        """Initialize the camera manager."""
        self.cap = None
    
    def connect(self):
        """
        Connect to the camera device.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            self.cap = cv2.VideoCapture(0)  # Try default camera first
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            
            ret, frame = self.cap.read()
            if not ret:
                raise RuntimeError("Could not read from camera")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS)
            
            logger.info(f"Camera connected with resolution {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS}fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to camera: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def is_connected(self):
        """
        Check if the camera is connected.
        
        Returns:
            bool: True if camera is connected, False otherwise.
        """
        return self.cap is not None and self.cap.isOpened()
    
    def get_frame(self):
        """
        Get the latest frame from the camera.
        
        Returns:
            numpy.ndarray: Video frame or None if error.
        """
        if not self.is_connected():
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret:
                return frame
            else:
                logger.warning("Failed to read from camera")
                return None
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def release(self):
        """Release the camera resources."""
        if self.cap:
            try:
                logger.debug("Releasing camera")
                self.cap.release()
                self.cap = None
            except Exception as e:
                logger.error(f"Error releasing camera: {e}") 