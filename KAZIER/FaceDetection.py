import cv2
import mediapipe as mp
import time
import math

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, static_mode=False, max_faces=2, min_tracking_confidence=0.5,flip=True):
        self.min_detection_confidence = min_detection_confidence
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.flip = flip
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=self.min_detection_confidence)
        
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=self.static_mode, 
                                                    max_num_faces=self.max_faces, 
                                                    min_detection_confidence=self.min_detection_confidence, 
                                                    min_tracking_confidence=self.min_tracking_confidence)
        self.draw_spec = self.mp_draw.DrawingSpec(thickness=2, circle_radius=2)

    def find_faces(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.flip:
            img = cv2.flip(img,1)
        self.results = self.face_detection.process(img_rgb)
        bboxs = []
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                if draw:
                    img = self.imp_draw(img, bbox)
                    cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 250), 2)
        return img, bboxs

    def imp_draw(self, img, bbox, l=30, t=10, rt=1): #it is to make square near face
        x, y, w, h = bbox
        x1, y1 = x + w, y + h
        cv2.rectangle(img, bbox, (255, 0, 0), rt)
        # Top left
        cv2.line(img, (x, y), (x + l, y), (255, 255, 0), 2)
        cv2.line(img, (x, y), (x, y + l), (255, 255, 0), 2)
        # Top right
        cv2.line(img, (x1, y), (x1 - l, y), (255, 255, 0), 2)
        cv2.line(img, (x1, y), (x1, y + l), (255, 255, 0), 2)
        # Bottom left
        cv2.line(img, (x, y1), (x + l, y1), (255, 255, 0), 2)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 255, 0), 2)
        # Bottom right
        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 255, 0), 2)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 255, 0), 2)
        return img

    def find_face_mesh(self, img, meshdraw=True,pointdraw=True):
        if self.flip:
            img = cv2.flip(img,1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(img_rgb)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if meshdraw:
                    self.mp_draw.draw_landmarks(img, faceLms, self.mp_face_mesh.FACEMESH_CONTOURS,
                                           self.draw_spec, self.draw_spec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id),(x,y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)
                    # print(id, x, y)
                    face.append([x, y])
                faces.append(face)
                #These are some main points using which we can work
                facial_landmarks = {
                'left_eye': [33, 133, 160, 159, 158, 157, 173, 153, 144, 163, 7, 246, 161, 160, 159, 158],
                'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
                'left_eyebrow': [70, 63, 105, 66, 107, 55, 193],
                'right_eyebrow': [336, 296, 334, 293, 300, 276, 283],
                'nose':[1, 2, 5, 4, 98, 97, 2, 327, 326, 193, 209, 198, 217, 275, 363, 456, 419, 331, 274, 275, 195, 193],
                'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
                'forehead': [10, 338, 297, 332, 284, 251, 389, 361, 454, 323, 361, 355, 368, 264, 447, 386, 374, 373, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 176, 148, 152, 377, 400, 378, 379, 397, 288]}           
                # Colors for each part
                colors = {
                    'left_eye': (0, 255, 0),
                    'right_eye': (255, 0, 0),
                    'left_eyebrow': (0, 255, 255),
                    'right_eyebrow': (255, 255, 0),
                    'nose': (255, 0, 255),
                    'mouth': (0, 0, 255),
                    'forehead': (128, 0, 128),
                }
                if pointdraw:
                    if len(faces) != 0:
                        for face in faces:
                            for part, points in facial_landmarks.items():
                                for point in points:
                                    coord = face[point]
                                    cv2.circle(img, (coord[0], coord[1]), 3, colors[part], cv2.FILLED)
                                    cv2.putText(img, str(point), (coord[0], coord[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return img, faces
    def findDistance(self,p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        if not success:
            break

        # Detect faces and draw bounding boxes using imp_draw
        img, bboxs = detector.find_faces(img)
        if bboxs:
            # Draw styled bounding boxes using imp_draw
            for _, bbox, _ in bboxs:
                img = detector.imp_draw(img, bbox)
            print("Bounding boxes:", bboxs)

        # Detect face mesh and draw landmarks
        img, faces = detector.find_face_mesh(img)
        if faces:
            print(f"Number of faces detected: {len(faces)}")

            # Example of finding distance between two landmarks for the first face detected
            p1 = faces[0][33]  # Example: left eye landmark
            p2 = faces[0][263] # Example: right eye landmark
            length, info, img = detector.findDistance(p1, p2, img)
            print(f"Distance between points: {length}, Info: {info}")

        # Display the resulting frame
        cv2.imshow('Image', img)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

