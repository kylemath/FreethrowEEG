import cv2
import time

def test_camera():
    print("Testing camera access...")
    
    # Try different camera indices
    for camera_index in [0, 1, 2]:
        print(f"\nTrying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        time.sleep(1)
        
        if cap.isOpened():
            print(f"Camera {camera_index} opened successfully")
            ret, frame = cap.read()
            if ret:
                print(f"Successfully read frame from camera {camera_index}")
                print(f"Frame shape: {frame.shape}")
                cap.release()
                return True
            else:
                print(f"Could not read frame from camera {camera_index}")
            cap.release()
        else:
            print(f"Could not open camera {camera_index}")
    
    return False

if __name__ == "__main__":
    success = test_camera()
    if not success:
        print("\nNo working cameras found. Please check your camera permissions:")
        print("1. Go to System Settings > Privacy & Security > Camera")
        print("2. Make sure Terminal has camera access")
        print("3. Try closing other applications that might be using the camera") 