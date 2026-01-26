#!/usr/bin/env python3
"""
Quick test script to verify Muse connection without the full GUI.
Run this to diagnose connection issues.
"""

import sys
import time
sys.path.insert(0, '.')

from src.utils.config import MUSE_DEVICES, DEFAULT_MUSE_DEVICE
from src.utils.logger import logger
from brainflow.board_shim import BoardShim, BrainFlowInputParams

def test_muse_connection(device_type=None):
    """Test connecting to a Muse device."""
    
    if device_type is None:
        device_type = DEFAULT_MUSE_DEVICE
    
    print("\n" + "=" * 60)
    print("FreethrowEEG - Muse Connection Test")
    print("=" * 60)
    
    print(f"\nAvailable Muse devices: {list(MUSE_DEVICES.keys())}")
    print(f"Testing connection to: {device_type}")
    
    if device_type not in MUSE_DEVICES:
        print(f"ERROR: Unknown device type. Choose from: {list(MUSE_DEVICES.keys())}")
        return False
    
    board_id = MUSE_DEVICES[device_type]
    print(f"BrainFlow board ID: {board_id}")
    
    try:
        params = BrainFlowInputParams()
        params.serial_port = ""  # Auto-discovery
        params.timeout = 15
        
        print("\nEnabling BrainFlow logger...")
        BoardShim.enable_dev_board_logger()
        
        print(f"Creating BoardShim for {device_type}...")
        board = BoardShim(board_id, params)
        
        print("Preparing session (this may take 10-15 seconds)...")
        print("Make sure your Muse is:")
        print("  1. Turned ON (hold power button)")
        print("  2. In pairing mode (light should be blinking)")
        print("  3. NOT connected to any other app (Mind Monitor, Muse app, etc.)")
        print()
        
        board.prepare_session()
        
        if not board.is_prepared():
            print("ERROR: Board preparation failed!")
            return False
        
        print("\nSUCCESS: Board session prepared!")
        
        # Get device info
        sampling_rate = BoardShim.get_sampling_rate(board_id)
        eeg_channels = BoardShim.get_eeg_channels(board_id)
        
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  EEG channels: {eeg_channels}")
        
        print("\nStarting data stream...")
        board.start_stream()
        
        print("Collecting 3 seconds of data...")
        time.sleep(3)
        
        data = board.get_board_data()
        print(f"\nCollected {data.shape[1]} samples")
        
        if data.shape[1] > 0:
            print("\nSample EEG values (first channel):")
            eeg_data = data[eeg_channels[0]][:10]
            print(f"  {eeg_data}")
            
            print("\n" + "=" * 60)
            print("CONNECTION TEST PASSED!")
            print("Your Muse is working correctly.")
            print("=" * 60)
        else:
            print("\nWARNING: No data received from device!")
        
        print("\nCleaning up...")
        board.stop_stream()
        board.release_session()
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Make sure Muse is turned on and in pairing mode")
        print("  2. Check Bluetooth is enabled on your computer")
        print("  3. Try removing and re-pairing the Muse in Bluetooth settings")
        print("  4. Make sure no other app is using the Muse")
        print("  5. Try a different device type if you have a different Muse model")
        return False


def list_devices():
    """List all supported Muse devices."""
    print("\nSupported Muse Devices:")
    print("-" * 40)
    for name, board_id in MUSE_DEVICES.items():
        marker = " (default)" if name == DEFAULT_MUSE_DEVICE else ""
        print(f"  {name}: board_id={board_id}{marker}")
    print()


if __name__ == "__main__":
    list_devices()
    
    # Get device type from command line if provided
    if len(sys.argv) > 1:
        device = sys.argv[1]
    else:
        device = None
        print(f"Using default device: {DEFAULT_MUSE_DEVICE}")
        print("To test a different device, run:")
        print('  python test_muse_connection.py "Muse S"')
        print()
    
    success = test_muse_connection(device)
    sys.exit(0 if success else 1)
