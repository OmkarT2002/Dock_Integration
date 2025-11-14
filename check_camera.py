# Setup camera — auto-detect first available index
import cv2
cap = None
for i in range(5):
    test_cap = cv2.VideoCapture(i)
    if test_cap.isOpened():
        print(f"✅ Camera found at index {i}")
        cap = test_cap
        break
    test_cap.release()

if cap is None or not cap.isOpened():
    raise RuntimeError("❌ Cannot open any available video source")
