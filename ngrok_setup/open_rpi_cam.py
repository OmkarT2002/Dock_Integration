import cv2

cap = cv2.VideoCapture("https://alla-unendeared-reed.ngrok-free.dev/video")

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame received — check that the Pi stream is running.")
        break
    cv2.imshow("Remote Pi Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
