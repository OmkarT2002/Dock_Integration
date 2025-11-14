import cv2
import numpy as np
from itertools import combinations

# Load camera calibration
camera_matrix = np.load('/run/media/ostajanpure/New Volume/Dock/takeoff_mechanism/camera_matrix.npy')
dist_coeffs = np.load('/run/media/ostajanpure/New Volume/Dock/takeoff_mechanism/dist_coeffs.npy')

# Dictionary: marker ID -> marker length (meters)
marker_sizes = {
    3: 0.067,
    0: 0.098,
    2: 0.049,
    1: 0.146
}

# ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

cap = cv2.VideoCapture(0)
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    h, w, _ = frame.shape
    cv2.circle(frame, (w//2, h//2), 4, (0, 0, 255), -1)

    corners, ids, _ = detector.detectMarkers(frame)

    marker_centers = {}

    if ids is not None:
        ids = ids.flatten()

        cv2.aruco.drawDetectedMarkers(frame, corners)

        for i, marker_id in enumerate(ids):

            if marker_id not in marker_sizes:
                print(f"Marker {marker_id} found but NO SIZE provided!")
                continue

            marker_length = marker_sizes[marker_id]

            # Pose estimation using each marker's own size
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[i]], marker_length, camera_matrix, dist_coeffs
            )

            rvec = rvec[0]
            tvec = tvec[0]

            marker_centers[marker_id] = tvec[0]

            # Draw axis
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                              rvec, tvec, marker_length)

            # Display text
            x, y, z = tvec[0]
            cv2.putText(frame, f"ID {marker_id} X:{x:.2f} Y:{y:.2f} Z:{z:.2f}",
                        (10, 40 + i * 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)

            print(f"Marker {marker_id}  -->  X:{x:.3f}  Y:{y:.3f}  Z:{z:.3f}")

        # Check whether all 4 markers detected
        if len(marker_centers) == 4:
            cv2.putText(frame, "ALL 4 MARKERS DETECTED", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            print("\n*** ALL 4 MARKERS DETECTED ***\n")

        # Compute distances between markers
        for (id1, id2) in combinations(marker_centers.keys(), 2):
            p1 = marker_centers[id1]
            p2 = marker_centers[id2]
            dist = np.linalg.norm(p1 - p2)

            print(f"Distance Marker {id1} <-> {id2}: {dist:.3f} m")

    cv2.imshow("Aruco Multi-Marker (Different Sizes)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
