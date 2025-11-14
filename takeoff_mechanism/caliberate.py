import cv2
import numpy as np
import glob
import os

# ---------- SETTINGS ----------
chessboard_size = (8,5)
square_size = 0.029  # meters per square (adjust to your printed size)
image_folder = r"/run/media/ostajanpure/New Volume/Dock/takeoff_mechanism/chess_board_images"  # raw string for path

# Prepare 3D points for chessboard pattern
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# ---------- LOAD IMAGES ----------
images = glob.glob(os.path.join(image_folder, "*.jpg"))  # change to *.png if needed

if len(images) == 0:
    print("❌ No images found! Check your path or file extension.")
    exit()

print(f"Found {len(images)} images for calibration...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Processing {fname}...")
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# ---------- CALIBRATE CAMERA ----------
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# ---------- SAVE RESULTS ----------
np.save('camera_matrix.npy', camera_matrix)
np.save('dist_coeffs.npy', dist_coeffs)

print("\n✅ Calibration successful!")
print("Camera matrix:\n", camera_matrix)
print("\nDistortion coefficients:\n", dist_coeffs)
