import cv2
import numpy as np

# Load camera calibration
camera_matrix = np.load('/run/media/ostajanpure/New Volume/Dock/takeoff_mechanism/camera_matrix.npy')
dist_coeffs = np.load('/run/media/ostajanpure/New Volume/Dock/takeoff_mechanism/dist_coeffs.npy')

# Marker side length in meters
#marker_length = 0.098  # 9.8 cm
marker_length = 0.067  # 6.7 cm
#marker_length = 0.049  # 4.9 cm
#marker_length = 0.146  # 14.6 cm
# Define ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Start video capture
cap = cv2.VideoCapture(2)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    h,w,c=frame.shape
    center = (w // 2, h // 2)  # (x, y)
    cv2.circle(frame, center, 2, (0, 0, 255), -1)
    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        # Estimate pose
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, marker_length, camera_matrix, dist_coeffs
        )

        for i in range(len(ids)):
            cv2.aruco.drawDetectedMarkers(frame, corners)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)

            # Extract translation (x, y, z)
            x, y, z = tvecs[i][0]

            # Display coordinates on screen
            text = f"X: {x:.3f} m  Y: {y:.3f} m  Z: {z:.3f} m"
            cv2.putText(frame, text, (10, 50 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Also print to console for logging/debug
            print(f"Marker ID {ids[i][0]}  ->  X: {x:.3f}, Y: {y:.3f}, Z: {z:.3f} (meters)")

    cv2.imshow('Aruco 3D Pose', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
