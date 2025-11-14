import cv2
import numpy as np

# ---------- SETTINGS ----------
marker_size = 0.098  # meters (actual marker side length, e.g., 10 cm)
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

# ---------- VIDEO SOURCE ----------
# 0 -> Default webcam. Replace with your drone feed or video file path if needed.
cap = cv2.VideoCapture(1)

# ---------- ARUCO SETUP ----------
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

print("🎥 Real-time ArUco height estimation started — press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Unable to access camera or video feed.")
        break

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Estimate pose (rotation + translation vectors)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_size, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            # Draw marker boundaries and 3D axis
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)

            # Extract Z distance (height above the marker plane)
            height_m = tvecs[i][0][2]

            # Overlay height text
            cv2.putText(frame, f"Height: {height_m:.2f} m", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            print(f"Marker ID {ids[i][0]} → Height: {height_m:.2f} m")

    # Show the frame
    cv2.imshow("Aruco Height Estimation", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
