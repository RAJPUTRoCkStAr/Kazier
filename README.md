# Kaizer Package

Kaizer is a comprehensive package for face detection, hand tracking, pose estimation, and more using MediaPipe. It is designed to simplify your project development.

## Features
- **Face Detection**: Efficient and accurate face detection.
- **Hand Tracking**: Real-time hand tracking and gesture recognition.
- **Pose Estimation**: Full-body pose estimation.
- **FPS Calculation**: Measure frames per second for performance evaluation.
- **Utilities**: Additional tools to streamline your project work.

## Installation

You can install the package using pip:

```bash
pip install kaizer
```

## Usage
### Using Face Detection
```bash
from KAZIER import FaceDetector 
import cv2

cap = cv2.VideoCapture(0)
detector = FaceDetector()
while True:
    success, img = cap.read()
    if not success:
        break
    img, bboxs = detector.find_faces(img)
    if bboxs:
        for _, bbox, _ in bboxs:
            img = detector.imp_draw(img, bbox)
        print("Bounding boxes:", bboxs)
    img, faces = detector.find_face_mesh(img)
    if faces:
        print(f"Number of faces detected: {len(faces)}")
        p1 = faces[0][33]  # Example: left eye landmark
        p2 = faces[0][263] # Example: right eye landmark
        length, info, img = detector.findDistance(p1, p2, img)
        print(f"Distance between points: {length}, Info: {info}")
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```


### Using fps
```bash
from KAZIER import FPS
import cv2

fps_counter = FPS()
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    fps = fps_counter.showfps(frame, writetext=True, text_pos=(10, 50),
                            fthickness=2,tcolor=(0,255,250),
                            Fstyle=cv2.FONT_HERSHEY_DUPLEX,fscale=2,)
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

### Using HAND DETECTION
```bash
from KAZIER import HandStar
import cv2

cap = cv2.VideoCapture(0)
detector = HandStar(maxHands=2)
while True:
    success, img = cap.read()
    if not success:
        break
    img = detector.detect_hands(img)
    lmList = detector.get_hand_positions(img)
    if len(lmList) != 0:
        fingersList = detector.get_fingers_status()
        for i, fingers in enumerate(fingersList):
            length, img, lineInfo = detector.calculate_distance(4, 8, img, handNo=i)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

### Using Pose Module
```bash
from KAZIER import PoseDetector
import cv2

cap = cv2.VideoCapture(0)
detector = PoseDetector()
while True:
    success, img = cap.read()
    img = cv2.resize(img, (680, 680))
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if lmList:
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 10, (250, 0, 0), cv2.FILLED)
        length, img, info = detector.findDistance(lmList[11][1:3], lmList[15][1:3], img=img, color=(255, 0, 0), scale=10)
    cv2.imshow("image", img)
    if cv2.waitKey(1) == ord('q'):
        break
```

### Using Utils
```bash
from KAZIER import Helper
import cv2

utils = Helper()
image_url = 'https://image.shutterstock.com/image-vector/dotted-spiral-vortex-royaltyfree-images-600w-2227567913.jpg'  # Replace with the actual image URL
image = utils.download_image_from_url(image_url)
black_background_image = utils.make_background_black(image)
rotated_image = utils.rotate_image(image, 45)
img2 = cv2.imread('med/ig.jpg')  
hstacked_image = utils.hstack_images(image, img2)
vstacked_image = utils.vstack_images(image, img2)
detected_color = utils.detect_color(image, 'green')
image_with_corners = utils.detect_corners(image)
image_with_text_left = utils.add_text(image, 'Hello World', (50, 50), font_name='hershey_triplex', color_name='blue', align='left')
cv2.waitKey(0)
cv2.destroyAllWindows()
```
## License
- This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing
- Contributions are welcome! Please open an issue or submit a pull request.

## Contact
- Replace `sumitsingh9441@gmail.com` with your actual email address. This `README.md` file now reflects the package name `kaizer` and includes usage examples for its features.
