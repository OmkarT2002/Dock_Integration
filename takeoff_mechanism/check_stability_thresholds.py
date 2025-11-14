import cv2
import numpy as np
import time
import math
import csv
from collections import deque
from statistics import mean, pstdev

# ---------------- CONFIGURATION ----------------
MODE = "indoor"  # "indoor" or "outdoor"

if MODE == "indoor":
    DEFLECTION_TOL = 0.10    # 10 cm tolerance
    WARNING_RMS = 0.05       # warning if RMS > 5 cm
    CRITICAL_RMS = 0.10      # critical if RMS > 10 cm
else:
    DEFLECTION_TOL = 0.30    # 30 cm tolerance (mid outdoor range)
    WARNING_RMS = 0.10       # warning if RMS > 10 cm
    CRITICAL_RMS = 0.20      # critical if RMS > 20 cm

MARKER_SIZE = 0.098  # marker side in meters (adjust to your printed size)
BUFFER_LEN = 60      # last N samples to compute RMS

# Load calibration results
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

# Setup camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# ArUco dictionary and detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

# Data buffers
times, xs, ys, zs, defs = deque(maxlen=BUFFER_LEN), deque(maxlen=BUFFER_LEN), deque(maxlen=BUFFER_LEN), deque(maxlen=BUFFER_LEN), deque(maxlen=BUFFER_LEN)

# CSV log
csv_file = open('stability_thresholds_log.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "x_m", "y_m", "z_m", "deflection_m", "RMS", "stability"])

print("🎥 ArUco Stability Monitor started — press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        # Assume single marker (first one)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, MARKER_SIZE, camera_matrix, dist_coeffs)
        (rvec, tvec) = (rvecs[0][0], tvecs[0][0])
        x, y, z = tvec[0], tvec[1], tvec[2]
        deflection = math.sqrt(x**2 + y**2)

        times.append(time.time())
        xs.append(x)
        ys.append(y)
        zs.append(z)
        defs.append(deflection)

        # Compute RMS over buffer
        rms = math.sqrt(sum(d*d for d in defs) / len(defs))

        # Determine stability status
        if rms <= WARNING_RMS:
            status = "STABLE"
            color = (0, 255, 0)
        elif rms <= CRITICAL_RMS:
            status = "WARNING"
            color = (0, 255, 255)
        else:
            status = "CRITICAL"
            color = (0, 0, 255)

        # Log data
        csv_writer.writerow([time.strftime("%H:%M:%S"), x, y, z, deflection, rms, status])
        csv_file.flush()

        # Draw marker and info
        cv2.aruco.drawDetectedMarkers(frame, corners)
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

        cv2.putText(frame, f"Deflection: {deflection:.3f} m", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"RMS: {rms:.3f} m", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    else:
        cv2.putText(frame, "Marker not detected!", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Drone Stability Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
