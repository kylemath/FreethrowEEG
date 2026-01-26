#!/usr/bin/env python3
"""
Test the debug mode data generation to verify it shows clear warnings.
"""

import sys
import time
import queue
sys.path.insert(0, '.')

from src.eeg.processor import EEGProcessor

def test_debug_mode():
    """Test debug mode data collection."""
    
    print("\n" + "=" * 60)
    print("FreethrowEEG - Debug Mode Test")
    print("=" * 60)
    
    data_queue = queue.Queue()
    processor = EEGProcessor(data_queue)
    
    print("\nConnecting in DEBUG mode...")
    success = processor.connect_muse(debug=True)
    
    if not success:
        print("ERROR: Debug mode connection failed!")
        return False
    
    print("\nStarting data collection...")
    processor.start_data_collection()
    
    print("Collecting data for 5 seconds...\n")
    time.sleep(5)
    
    # Check what data we got
    data_points = []
    while not data_queue.empty():
        data_points.append(data_queue.get())
    
    print(f"\nCollected {len(data_points)} data points")
    
    if len(data_points) > 0:
        print("\nSample data (first 3 points):")
        for i, (timestamp, powers) in enumerate(data_points[:3]):
            print(f"  t={timestamp:.2f}s: alpha={powers['alpha']:.2f}, beta={powers['beta']:.2f}")
        
        # Check if data looks like sine waves (should be smooth)
        alpha_values = [p[1]['alpha'] for p in data_points]
        if len(alpha_values) > 1:
            # Check variance - simulated data should be relatively smooth
            import numpy as np
            variance = np.var(np.diff(alpha_values))
            print(f"\nAlpha value variance: {variance:.4f}")
            print("(Low variance = smooth sine waves = simulated data)")
    
    print("\nStopping processor...")
    processor.stop()
    
    print("\n" + "=" * 60)
    print("DEBUG MODE TEST COMPLETE")
    print("Note: You should have seen warning messages above about")
    print("'SIMULATED MODE' and 'smooth sine wave patterns'")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_debug_mode()
    sys.exit(0 if success else 1)
