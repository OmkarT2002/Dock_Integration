from ultralytics import YOLO
import cv2
import math
import time
import csv
import os
from datetime import datetime

# ------------------- SETTINGS -------------------
MODEL_PATH = r"D:\Dock\Dock_door\dock_light_2_n.pt"
CSV_FILE = "D:\Dock\Dock_door\door_events.csv"
LOG_FILE = "D:\Dock\Dock_door\door_log.txt"
DOOR_STATUS_FILE = "D:\Dock\Dock_door\door_status.txt"
DISTANCE_THRESHOLD = 49  # px
CONF_THRESHOLD = 0.5
CHECK_DELAY = 13  # seconds after both lights detected
# ------------------------------------------------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Error: Cannot open camera.")
    exit()

print("✅ Starting real-time detection... Press 'q' to quit.")

# Make CSV if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["entry_time", "closing_time"])

CLASS_NAMES = {0: "green", 1: "red"}

# Tracking variables
entry_times = {"green": None, "red": None}
prev_centers = {"green": None, "red": None}
direction_detected = {"green": False, "red": False}
door_closed_logged = False
check_scheduled_time = None  # time to check door closure
check_ready = False  # flag for delayed check

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    frame_width = frame.shape[1]
    results = model.predict(source=frame, conf=CONF_THRESHOLD, show=False, verbose=False)
    boxes = results[0].boxes
    centers = {}

    # ------------------- DETECTION LOOP -------------------
    for box in boxes:
        cls_id = int(box.cls[0])
        if cls_id in CLASS_NAMES:
            label = CLASS_NAMES[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            centers[label] = (cx, cy)

            color = (0, 255, 0) if label == "green" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)

            # Track motion
            prev_cx = prev_centers[label]
            if prev_cx is not None:
                movement = cx - prev_cx

                # Green enters left→right
                if label == "green" and not direction_detected["green"]:
                    if prev_cx < frame_width * 0.2 and movement > 0:
                        entry_times["green"] = time.time()
                        direction_detected["green"] = True
                        door_closed_logged = False
                        print(f"🟢 Green entered at {entry_times['green']:.2f}s")

                # Red enters right→left
                if label == "red" and not direction_detected["red"]:
                    if prev_cx > frame_width * 0.8 and movement < 0:
                        entry_times["red"] = time.time()
                        direction_detected["red"] = True
                        door_closed_logged = False
                        print(f"🔴 Red entered at {entry_times['red']:.2f}s")

            prev_centers[label] = cx

    # ------------------- DISTANCE DISPLAY -------------------
    distance = None
    if "green" in centers and "red" in centers:
        (x1, y1), (x2, y2) = centers["green"], centers["red"]
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # ------------------- ENTRY LOGIC -------------------
    if entry_times["green"] and entry_times["red"] and not door_closed_logged:
        time_diff = abs(entry_times["green"] - entry_times["red"])
        current_time = time.time()

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Log delay or success
        if time_diff > 5:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] ❌ ERROR 101: Dock detected delay in door opening ({time_diff:.2f}s)\n")
            print("⚠️ ERROR 101 logged")
        else:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] ✅ SUCCESS: Door initiation normal (Δt={time_diff:.2f}s)\n")
            print("✅ Door initiation success logged")

        # Schedule door check after 13 seconds
        check_scheduled_time = current_time + CHECK_DELAY
        check_ready = True
        door_closed_logged = True

        # Log in CSV
        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"{timestamp}", f"{time_diff:.2f}s"])

    # ------------------- DOOR STATUS CHECK -------------------
    if check_ready and time.time() >= check_scheduled_time:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("⏳ Waiting 13 seconds before checking door status...")
        time.sleep(1)  # wait 13s after initiation

        # Capture a fresh frame
        ret_after, frame_after = cap.read()
        if not ret_after:
            print("❌ Failed to grab frame for 13s check.")
            with open(DOOR_STATUS_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] ⚠️ ERROR: Could not capture frame after 13s.\n")
            check_ready = False
            continue

        # Run YOLO again on the new frame
        results_after = model.predict(source=frame_after, conf=CONF_THRESHOLD, show=False, verbose=False)
        boxes_after = results_after[0].boxes
        centers_after = {}

        for box in boxes_after:
            cls_id = int(box.cls[0])
            if cls_id in CLASS_NAMES:
                label = CLASS_NAMES[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                centers_after[label] = (cx, cy)

        # Calculate distance after 13 seconds
        distance_after = None
        if "green" in centers_after and "red" in centers_after:
            (x1, y1), (x2, y2) = centers_after["green"], centers_after["red"]
            distance_after = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            print(f"📏 Distance after 13s: {distance_after:.2f}px")

        # Log door closure result
        with open(DOOR_STATUS_FILE, "a", encoding="utf-8") as f:
            if distance_after is not None:
                if distance_after < DISTANCE_THRESHOLD:
                    f.write(f"[{timestamp}] ✅ Door closed properly (distance: {distance_after:.2f}px)\n")
                    print("✅ Door closed properly logged")
                else:
                    f.write(f"[{timestamp}] ❌ Door not closed properly (distance: {distance_after:.2f}px)\n")
                    print("⚠️ Door not closed properly logged")
            else:
                f.write(f"[{timestamp}] ⚠️ ERROR: Could not detect both lights after 13s.\n")
                print("⚠️ Could not detect both lights after 13 seconds.")

        # Reset for next cycle
        check_ready = False
        entry_times = {"green": None, "red": None}
        direction_detected = {"green": False, "red": False}
        prev_centers = {"green": None, "red": None}

    # ------------------- DISPLAY -------------------
    info_text = ""
    if distance is not None:
        info_text += f"Distance: {distance:.2f}px"
    if entry_times["green"] and entry_times["red"]:
        time_diff = abs(entry_times["green"] - entry_times["red"])
        info_text += f" | Δt: {time_diff:.2f}s"

    if info_text:
        cv2.putText(frame, info_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 | Distance + Time + CSV Logging", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------- CLEANUP -------------------
cap.release()
cv2.destroyAllWindows()
print("✅ Detection stopped.")
