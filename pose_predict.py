import cv2
import math
from ultralytics import YOLO

def calculate_angle(pivot, tip):
    """
    Calculate angle of link relative to horizontal.
    pivot, tip: (x, y)
    Returns angle in degrees.
    """
    dx = tip[0] - pivot[0]
    dy = tip[1] - pivot[1]
    angle = math.degrees(math.atan2(-dy, dx))  # Negative because y increases downward
    return angle

def main(model_path='D:\Dock\side_link_pose.pt', source=0):
    # Load trained YOLOv8 pose model
    model = YOLO(model_path)

    # Start camera or video stream
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Cannot open camera or video source: {source}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ No frame received. Stream may have ended.")
            break

        # Run YOLOv8 pose inference
        results = model(frame, verbose=False)

        for r in results:
            keypoints = r.keypoints
            if keypoints is not None and len(keypoints.xy) > 0:
                pts = keypoints.xy[0].cpu().numpy()  # first detected object
                if len(pts) >= 2:
                    pivot = (int(pts[0][0]), int(pts[0][1]))
                    tip = (int(pts[1][0]), int(pts[1][1]))

                    # Calculate the link’s rotation angle
                    angle = calculate_angle(pivot, tip)

                    # Draw keypoints and line
                    cv2.circle(frame, pivot, 6, (0, 255, 0), -1)
                    cv2.circle(frame, tip, 6, (0, 0, 255), -1)
                    cv2.line(frame, pivot, tip, (255, 255, 0), 2)
                    cv2.putText(frame, f"Angle: {angle:.1f} deg", (pivot[0] + 20, pivot[1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Show output frame
        cv2.imshow('Pose Detection - press q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change `source=0` to your camera index or video path, e.g. 'test.mp4'
    main(model_path='D:\Dock\side_link_pose1.pt', source=1)
