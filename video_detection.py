from ultralytics import YOLO
import cv2

# Load fire and smoke model
model = YOLO("models/best.pt")

# Video path
video_path = "test_videos/test.mp4"

# Open video
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run detection on each frame
    results = model(frame)

    for result in results:
        output = result.plot()

        cv2.imshow("Fire Smoke Detection", output)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()