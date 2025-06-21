import cv2

# Load the Haar cascade classifier from OpenCV's built-in path
haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Try all webcam indexes (0, 1, 2...) — use 0 by default
cam = cv2.VideoCapture(0)

# Check if camera is opened
if not cam.isOpened():
    print("❌ Error: Cannot access the webcam. Try a different index (1, 2, etc).")
    exit()

print("✅ Webcam started. Press ESC to exit.")

while True:
    ret, img = cam.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Face Detection", img)

    # Press ESC (key code 27) to break
    if cv2.waitKey(1) & 0xFF == 27:
        print("👋 Exiting...")
        break

cam.release()
cv2.destroyAllWindows()
