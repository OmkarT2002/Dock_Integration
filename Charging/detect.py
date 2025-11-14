from ultralytics import YOLO
import cv2
import time
import datetime

# ---------------- SETTINGS ----------------
MODEL_PATH = r"D:\Dock\Charging\drone_light.pt"
VIDEO_PATH = r"D:\Dock\Charging\drone charge.mp4"
OUTPUT_PATH = r"output.mp4"
LOG_FILE = r"detections.log"
CONF_THRESHOLD = 0.2
TIME_WINDOW = 15  # seconds
IOU_THRESHOLD = 0.5  # For removing duplicate boxes

# ---------------- HELPER FUNCTIONS ----------------
def iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0

def remove_duplicates(boxes):
    """Remove overlapping boxes with high IoU."""
    filtered = []
    for box in boxes:
        if not any(iou(box, fb) > IOU_THRESHOLD for fb in filtered):
            filtered.append(box)
    return filtered

def count_line_overlaps(boxes):
    """For each box, draw a line through its center and count overlaps."""
    total_counts = []
    for box in boxes:
        x1, y1, x2, y2 = box
        y_center = (y1 + y2) / 2
        count = 0
        for other in boxes:
            ox1, oy1, ox2, oy2 = other
            if oy1 <= y_center <= oy2:
                count += 1
        total_counts.append(count)
    return max(total_counts) if total_counts else 0

# ---------------- LOAD MODEL ----------------
model = YOLO(MODEL_PATH)

# ---------------- OPEN VIDEO ----------------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (frame_width, frame_height)
)

# ---------------- MAIN LOOP ----------------
print("🚀 Processing started... Press 'q' to quit.")
frame_counts = []
window_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.xyxy is not None else []
    boxes = [list(map(int, box)) for box in detections]

    # Remove duplicates
    boxes = remove_duplicates(boxes)

    # Count overlaps with horizontal line through center
    count = count_line_overlaps(boxes)
    frame_counts.append(count)

    # Draw boxes and center lines
    for box in boxes:
        x1, y1, x2, y2 = box
        y_center = int((y1 + y2) / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.line(frame, (0, y_center), (frame.shape[1], y_center), (255, 0, 0), 1)

    # Display and save annotated frame
    cv2.imshow("YOLO Live Prediction", frame)
    out.write(frame)

    # ---------------- LOGGING SECTION ----------------
    current_time = time.time()
    elapsed = current_time - window_start

    if elapsed >= TIME_WINDOW:
        # Compute max count in the last 15 seconds
        max_count = max(frame_counts) if frame_counts else 0

        # Log it with timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a") as log:
            log.write(f"{timestamp} - Max count in last 15s: {max_count}\n")
        print(f"🕒 Logged {max_count} at {timestamp}")

        # Keep roughly last 15 seconds worth of counts to maintain rolling window
        frame_counts = frame_counts[-int(fps * TIME_WINDOW):]
        window_start = current_time

    # Quit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Done! Results saved to {OUTPUT_PATH}, log written to {LOG_FILE}")
