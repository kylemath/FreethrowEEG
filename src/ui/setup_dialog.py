"""
Setup dialog for the FreethrowEEG application.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from src.utils.logger import logger

class SetupDialog:
    """Handles the initial setup dialog for the application."""
    
    def __init__(self, on_connect_muse, on_connect_camera, on_start_session, on_check_ready):
        """
        Initialize the setup dialog.
        
        Args:
            on_connect_muse (callable): Callback for MUSE connection.
            on_connect_camera (callable): Callback for camera connection.
            on_start_session (callable): Callback for session start.
            on_check_ready (callable): Callback to check if setup is complete.
        """
        self.on_connect_muse = on_connect_muse
        self.on_connect_camera = on_connect_camera
        self.on_start_session = on_start_session
        self.on_check_ready = on_check_ready
        self.setup_fig = None
        self.player_box = None
        self.shots_box = None
        self.muse_status = None
        self.muse_button = None
        self.debug_button = None
        self.camera_status = None
        self.camera_button = None
        self.start_button = None
    
    def show(self):
        """Show the setup dialog."""
        self.setup_fig = plt.figure(figsize=(8, 6))
        self.setup_fig.canvas.manager.set_window_title('FreethrowEEG Setup')
        
        # Add text boxes for player info
        plt.figtext(0.1, 0.8, 'Player Name/ID:', size=10)
        self.player_box = TextBox(plt.axes([0.4, 0.8, 0.4, 0.05]), '', initial='001')
        
        plt.figtext(0.1, 0.7, 'Number of Shots:', size=10)
        self.shots_box = TextBox(plt.axes([0.4, 0.7, 0.4, 0.05]), '', initial='2')
        
        # Add status indicators and connect buttons
        self.muse_status = plt.figtext(0.1, 0.5, 'X MUSE Not Connected', color='red')
        
        # Create two buttons side by side for MUSE connection
        self.muse_button = Button(plt.axes([0.4, 0.5, 0.25, 0.05]), 'Connect MUSE')
        self.muse_button.on_clicked(lambda x: self._connect_muse_handler(x, debug=False))
        
        self.debug_button = Button(plt.axes([0.7, 0.5, 0.25, 0.05]), 'Debug Mode')
        self.debug_button.on_clicked(lambda x: self._connect_muse_handler(x, debug=True))
        
        self.camera_status = plt.figtext(0.1, 0.4, 'X Camera Not Connected', color='red')
        self.camera_button = Button(plt.axes([0.6, 0.4, 0.2, 0.05]), 'Connect Camera')
        self.camera_button.on_clicked(self._connect_camera_handler)
        
        self.start_button = Button(plt.axes([0.35, 0.2, 0.3, 0.1]), 'Start Session')
        self.start_button.on_clicked(self._start_session_handler)
        self.start_button.set_active(False)  # Disable until setup complete
        
        plt.show()
    
    def _connect_muse_handler(self, event, debug=False):
        """Handle MUSE connection button events."""
        if debug:
            self.muse_status.set_text('* Connecting to MUSE (Debug)...')
        else:
            self.muse_status.set_text('* Connecting to MUSE...')
        self.muse_status.set_color('orange')
        self.muse_button.set_active(False)
        self.debug_button.set_active(False)
        self.muse_button.label.set_text('Connecting...')
        plt.pause(0.01)  # Force update
        
        # Call the actual connection handler
        success = self.on_connect_muse(debug)
        
        if success:
            if debug:
                self.muse_status.set_text('✓ MUSE Simulated')
                self.muse_status.set_color('blue')
            else:
                self.muse_status.set_text('✓ MUSE Connected')
                self.muse_status.set_color('green')
            
            self.muse_button.label.set_text('Connected')
            
            # Disable the appropriate buttons
            if debug:
                self.muse_button.set_active(False)
                self.debug_button.label.set_text('Debug Active')
                self.debug_button.set_active(False)
            else:
                self.debug_button.set_active(False)
        else:
            self.muse_status.set_text('X MUSE Connection Failed')
            self.muse_status.set_color('red')
            self.muse_button.label.set_text('Retry Connect')
            self.muse_button.set_active(True)
            self.debug_button.set_active(True)
        
        plt.pause(0.01)  # Force update
        self.on_check_ready()
    
    def _connect_camera_handler(self, event):
        """Handle camera connection button events."""
        self.camera_status.set_text('* Connecting to Camera...')
        self.camera_status.set_color('orange')
        self.camera_button.set_active(False)
        self.camera_button.label.set_text('Connecting...')
        plt.pause(0.01)  # Force update
        
        # Call the actual connection handler
        success = self.on_connect_camera()
        
        if success:
            self.camera_status.set_text('✓ Camera Connected')
            self.camera_status.set_color('green')
            self.camera_button.label.set_text('Connected')
        else:
            self.camera_status.set_text('X Camera Connection Failed')
            self.camera_status.set_color('red')
            self.camera_button.label.set_text('Retry Connect')
            self.camera_button.set_active(True)
        
        plt.pause(0.01)  # Force update
        self.on_check_ready()
    
    def _start_session_handler(self, event):
        """Handle start session button events."""
        try:
            player_id = self.player_box.text.strip()
            num_shots = int(self.shots_box.text.strip())
            
            if len(player_id) == 0:
                logger.warning("Player ID is empty")
                return
            
            if num_shots <= 0:
                logger.warning("Invalid number of shots")
                return
            
            logger.info(f"Starting session with player {player_id}, {num_shots} shots")
            self.on_start_session(player_id, num_shots)
            plt.close(self.setup_fig)
            
        except Exception as e:
            logger.error(f"Error starting session: {e}")
    
    def update_ready_status(self, player_ready, shots_ready, muse_ready, camera_ready):
        """
        Update the ready status of the setup.
        
        Args:
            player_ready (bool): Whether player info is ready.
            shots_ready (bool): Whether shots info is ready.
            muse_ready (bool): Whether MUSE is ready.
            camera_ready (bool): Whether camera is ready.
        """
        all_ready = player_ready and shots_ready and muse_ready and camera_ready
        self.start_button.set_active(all_ready)
        
        if all_ready:
            logger.info("All setup complete - ready to start session")
        
        plt.pause(0.01)  # Force update
    
    def get_player_id(self):
        """Get the player ID from the text box."""
        return self.player_box.text.strip()
    
    def get_num_shots(self):
        """Get the number of shots from the text box."""
        try:
            return int(self.shots_box.text.strip())
        except ValueError:
            return 0 