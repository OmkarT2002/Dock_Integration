"""
aruco_stability_live_1s.py
Real-time ArUco-based drone stability monitor (OpenCV ≥ 4.7)
→ Plots 1 point per second in the 3D stability graph.
"""

import cv2
import numpy as np
import time
import csv
from collections import deque
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# ====== CONFIG ======
MARKER_SIZE_M = 0.098          # marker size in meters (9.8 cm)
CAMERA_MTX_FILE = 'camera_matrix.npy'
DIST_COEFFS_FILE = 'dist_coeffs.npy'
VIDEO_SOURCE = 2               # 0 for webcam
LOG_CSV = 'aruco_stability_log.csv'
BUFFER_LEN = 1000

# ====== LOAD CALIBRATION ======
camera_matrix = np.load(CAMERA_MTX_FILE)
dist_coeffs = np.load(DIST_COEFFS_FILE)

# ====== ARUCO SETUP (new API) ======
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# ====== DATA BUFFERS ======
times, xs, ys, zs = deque(maxlen=BUFFER_LEN), deque(maxlen=BUFFER_LEN), deque(maxlen=BUFFER_LEN), deque(maxlen=BUFFER_LEN)

# ====== CSV LOG ======
csv_file = open(LOG_CSV, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['timestamp', 'x_m', 'y_m', 'z_m', 'deflection_m', 'angle_deg'])

# ====== MATPLOTLIB LIVE PLOT ======
plt.ion()
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Drone stability (1 point/sec)')

# ====== VIDEO CAPTURE ======
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open video source")

print("🎥 Starting ArUco stability monitor (1 point/sec). Press 'q' to quit.")

last_plot_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No frame captured.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, MARKER_SIZE_M, camera_matrix, dist_coeffs
            )

            rvec = rvecs[0][0]
            tvec = tvecs[0][0].reshape(3, 1)

            # Convert to camera position in marker frame
            R_ca, _ = cv2.Rodrigues(rvec)
            cam_pos_marker = -R_ca.T.dot(tvec).flatten()
            x_m, y_m, z_m = cam_pos_marker.tolist()

            deflection = float(np.sqrt(x_m**2 + y_m**2))
            angle_deg = float(np.degrees(np.arctan2(deflection, max(z_m, 1e-6))))

            tnow = time.time()
            times.append(tnow)
            xs.append(x_m)
            ys.append(y_m)
            zs.append(z_m)
            csv_writer.writerow([tnow, x_m, y_m, z_m, deflection, angle_deg])
            csv_file.flush()

            # Draw marker axes
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, MARKER_SIZE_M * 0.5)

            cv2.putText(frame, f"x={x_m:.3f} y={y_m:.3f} z={z_m:.3f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"def={deflection:.3f}m ang={angle_deg:.2f}°",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 🕐 Update plot every 1 second
            if time.time() - last_plot_time >= 1.0:
                ax.clear()
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                ax.set_title('Drone stability (1 point/sec)')
                ax.scatter(xs, ys, zs, c='b', s=20)
                ax.plot(xs, ys, zs, linewidth=0.7, alpha=0.7)
                margin = 0.1
                ax.set_xlim(min(xs) - margin, max(xs) + margin)
                ax.set_ylim(min(ys) - margin, max(ys) + margin)
                ax.set_zlim(max(0.0, min(zs) - margin), max(zs) + margin)
                plt.draw()
                plt.pause(0.001)
                last_plot_time = time.time()

        cv2.imshow("ArUco Stability Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass
finally:
    csv_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Finished. Log saved to", LOG_CSV)
