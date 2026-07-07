from ultralytics import YOLO
import cv2

# Load the YOLO model
model = YOLO("models/best.pt")

# Test image path
image_path = "test_images/test.jpg"

# Run detection
results = model(image_path)

# Display result
for result in results:
    output = result.plot()

    cv2.imshow("Fire Smoke Detection", output)
    cv2.waitKey(0)

cv2.destroyAllWindows()