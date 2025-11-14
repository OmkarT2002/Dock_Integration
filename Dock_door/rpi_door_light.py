from ultralytics import YOLO
import cv2
import math
import time
import csv
import os
from datetime import datetime


class DoorMonitor:
    def __init__(self,
                 model_path,
                 csv_file,
                 log_file,
                 door_status_file,
                 distance_threshold=49,
                 conf_threshold=0.5,
                 check_delay=13,
                 camera_index=1):
        # ---------------- SETTINGS ----------------
        self.model = YOLO(model_path)
        self.csv_file = csv_file
        self.log_file = log_file
        self.door_status_file = door_status_file
        self.distance_threshold = distance_threshold
        self.conf_threshold = conf_threshold
        self.check_delay = check_delay
        self.cap = cv2.VideoCapture(r"/run/media/ostajanpure/New Volume/Dock/Dock_door/WIN_20251107_11_33_35_Pro.mp4")
        self.class_names = {0: "green", 1: "red"}

        # ---------------- STATE VARIABLES -----------------0
        self.entry_times = {"green": None, "red": None}
        self.prev_centers = {"green": None, "red": None}
        self.direction_detected = {"green": False, "red": False}
        self.door_closed_logged = False
        self.check_scheduled_time = None
        self.check_ready = False

        if not self.cap.isOpened():
            raise RuntimeError("Error: Cannot open camera.")

        # Make CSV if not exists
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["entry_time", "closing_time"])

        print("DoorMonitor initialized successfully.")

    # -----------------------------------------------------------
    def detect_objects(self, frame):
        results = self.model.predict(source=frame, conf=self.conf_threshold, show=False, verbose=False)
        boxes = results[0].boxes
        centers = {}

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.class_names:
                label = self.class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                centers[label] = (cx, cy)

        return centers, boxes

    # -----------------------------------------------------------
    def draw_detections(self, frame, boxes, centers):
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in self.class_names:
                label = self.class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                color = (0, 255, 0) if label == "green" else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cx, cy = centers[label]
                cv2.circle(frame, (cx, cy), 5, color, -1)
        return frame

    # -----------------------------------------------------------
    def process_movement(self, centers, frame_width):
        for label in centers.keys():
            cx, _ = centers[label]
            prev_cx = self.prev_centers[label]
            if prev_cx is not None:
                movement = cx - prev_cx

                # Green enters left→right
                if label == "green" and not self.direction_detected["green"]:
                    if prev_cx < frame_width * 0.2 and movement > 0:
                        self.entry_times["green"] = time.time()
                        self.direction_detected["green"] = True
                        self.door_closed_logged = False
                        print(f"Green entered at {self.entry_times['green']:.2f}s")

                # Red enters right→left
                if label == "red" and not self.direction_detected["red"]:
                    if prev_cx > frame_width * 0.8 and movement < 0:
                        self.entry_times["red"] = time.time()
                        self.direction_detected["red"] = True
                        self.door_closed_logged = False
                        print(f"Red entered at {self.entry_times['red']:.2f}s")

            self.prev_centers[label] = cx

    # -----------------------------------------------------------
    def calculate_distance(self, centers):
        if "green" in centers and "red" in centers:
            (x1, y1), (x2, y2) = centers["green"], centers["red"]
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return None

    # -----------------------------------------------------------
    def log_entry(self):
        if self.entry_times["green"] and self.entry_times["red"] and not self.door_closed_logged:
            time_diff = abs(self.entry_times["green"] - self.entry_times["red"])
            current_time = time.time()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if time_diff > 5:
                self._append_log(self.log_file, f"[{timestamp}] ERROR 101: Dock detected delay in door opening ({time_diff:.2f}s)\n")
                print("ERROR 101 logged")
            else:
                self._append_log(self.log_file, f"[{timestamp}] SUCCESS: Door initiation normal (Δt={time_diff:.2f}s)\n")
                print("Door initiation success logged")

            # Schedule 13s check
            self.check_scheduled_time = current_time + self.check_delay
            self.check_ready = True
            self.door_closed_logged = True

            # Log in CSV
            with open(self.csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, f"{time_diff:.2f}s"])

    # -----------------------------------------------------------
    def check_door_status(self):
        if self.check_ready and time.time() >= self.check_scheduled_time:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print("⏳ Performing door closure check...")

            ret, frame_after = self.cap.read()
            if not ret:
                self._append_log(self.door_status_file, f"[{timestamp}] ERROR: Could not capture frame after delay.\n")
                self.check_ready = False
                return

            centers_after, _ = self.detect_objects(frame_after)
            distance_after = self.calculate_distance(centers_after)

            if distance_after is not None:
                if distance_after < self.distance_threshold:
                    msg = f"[{timestamp}] Door closed properly (distance: {distance_after:.2f}px)\n"
                    print("Door closed properly logged")
                else:
                    msg = f"[{timestamp}] Door not closed properly (distance: {distance_after:.2f}px)\n"
                    print("Door not closed properly logged")
            else:
                msg = f"[{timestamp}] ERROR: Could not detect both lights after check.\n"
                print("Could not detect both lights after delay.")

            self._append_log(self.door_status_file, msg)
            self._reset_cycle()

    # -----------------------------------------------------------
    def _append_log(self, file_path, text):
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(text)

    # -----------------------------------------------------------
    def _reset_cycle(self):
        self.check_ready = False
        self.entry_times = {"green": None, "red": None}
        self.direction_detected = {"green": False, "red": False}
        self.prev_centers = {"green": None, "red": None}

    # -----------------------------------------------------------
    def run(self):
        print("Starting real-time detection... Press 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(" Failed to grab frame.")
                break

            frame_width = frame.shape[1]
            centers, boxes = self.detect_objects(frame)
            self.process_movement(centers, frame_width)
            distance = self.calculate_distance(centers)
            self.log_entry()
            self.check_door_status()

            # Draw results
            frame = self.draw_detections(frame, boxes, centers)
            info_text = ""
            if distance:
                info_text += f"Distance: {distance:.2f}px"
            if self.entry_times["green"] and self.entry_times["red"]:
                time_diff = abs(self.entry_times["green"] - self.entry_times["red"])
                info_text += f" | Δt: {time_diff:.2f}s"

            if info_text:
                cv2.putText(frame, info_text, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow("YOLOv8 | Distance + Time + CSV Logging", frame)
            # --- Stop condition if too close ---
            if distance is not None and distance < self.distance_threshold:
                print(f"🚪 Object within threshold ({distance:.2f}px < {distance_threshold}px). Stopping detection.")
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")


# -----------------------------------------------------------
if __name__ == "__main__":
    monitor = DoorMonitor(
        model_path=r"/run/media/ostajanpure/New Volume/Dock/Dock_door/best.pt",
        csv_file=r"/run/media/ostajanpure/New Volume/Dock/Dock_door/door_events.csv",
        log_file=r"/run/media/ostajanpure/New Volume/Dock/Dock_door/door_log.txt",
        door_status_file=r"/run/media/ostajanpure/New Volume/Dock/Dock_door/door_status.txt",
        distance_threshold=49,
        conf_threshold=0.5,
        check_delay=13,
        camera_index=0
    )
    monitor.run()
