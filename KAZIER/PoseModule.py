import cv2
import mediapipe as mp 
import time
import math

class PoseDetector:
    def __init__(self, mode=False, smooth=True, detectionCon=0.5, trackCon=0.5, flip=True):
        self.mode = mode
        self.smooth = smooth
        self.flip = flip
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(static_image_mode=self.mode, 
                                     smooth_landmarks=self.smooth, 
                                     min_detection_confidence=self.detectionCon, 
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
        if self.flip:
            img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        if self.flip:
            img = cv2.flip(img, 1)
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True, anglewrite=True):
        if len(self.lmList) != 0:
            _, x1, y1 = self.lmList[p1]
            _, x2, y2 = self.lmList[p2]
            _, x3, y3 = self.lmList[p3]
            angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
            if angle < 0:
                angle += 360
            if anglewrite:
                cv2.putText(img, str(int(angle)), (x2 - 100, y2 + 100), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 0, 255), 2)
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
                cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), 2)
                cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (0, 255, 0), 2)
                cv2.circle(img, (x3, y3), 10, (0, 255, 0), cv2.FILLED)
                cv2.circle(img, (x3, y3), 15, (0, 255, 0), 2)
        return angle

    def findDistance(self, p1, p2, img=None, color=(255, 0, 255), scale=5):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)

        if img is not None:
            cv2.line(img, (x1, y1), (x2, y2), color, max(1, scale // 3))
            cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
            cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
            cv2.circle(img, (cx, cy), scale, color, cv2.FILLED)

        return length, img, info
    
def main():
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

if __name__ == '__main__':
    main()
