import cv2
import time
import os

# --- Settings ---
save_dir = "/run/media/ostajanpure/New Volume/Dock/takeoff_mechanism/chess_board_images"
camera_index = 2  # change if your camera is not index 0
capture_interval = 1  # seconds between captures

# --- Create directory if not exists ---
os.makedirs(save_dir, exist_ok=True)

# --- Start video capture ---
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("❌ Cannot open camera.")
    exit()

print("📸 Starting capture — press 'q' to quit.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Save one frame per second
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"image_{frame_count:03d}_{timestamp}.jpg"
    filepath = os.path.join(save_dir, filename)
    cv2.imwrite(filepath, frame)
    print(f"✅ Saved {filepath}")

    frame_count += 1

    # Wait 1 second
    time.sleep(capture_interval)

    # Optional: show preview
    cv2.imshow("Live Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Capture stopped.")
