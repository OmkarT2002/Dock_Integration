import cv2

def main(source=2):
    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)  # Use DirectShow explicitly
    if not cap.isOpened():
        print(f"Cannot open camera/source: {source}")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream ended?). Exiting ...")
            break

        cv2.imshow('Camera Stream - press q to quit', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(2)  # or try 1 if you have multiple cameras
