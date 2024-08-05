import cv2
import mediapipe as mp
import time
import math

class HandStar:
    def __init__(self, mode=False, maxHands=2, detectioncon=0.5, trackcon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectioncon = detectioncon
        self.trackcon = trackcon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectioncon,
                                        min_tracking_confidence=self.trackcon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def detect_hands(self, img, draw=True,flip=True):
        if flip:
            img = cv2.flip(img, 1)
        imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgrgb)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def get_hand_positions(self, img, draw=True):
        self.lmList = []
        self.handTypes = []
        if self.results.multi_hand_landmarks:
            for handNo, handLms in enumerate(self.results.multi_hand_landmarks):
                handLmList = []
                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    handLmList.append([id, cx, cy])
                self.lmList.append(handLmList)
                if self.results.multi_handedness:
                    handType = self.results.multi_handedness[handNo].classification[0].label
                    self.handTypes.append(handType)
                    if draw and handLmList:
                        x_min = min([point[1] for point in handLmList])
                        y_min = min([point[2] for point in handLmList])
                        x_max = max([point[1] for point in handLmList])
                        y_max = max([point[2] for point in handLmList])
                        cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
                        cv2.putText(img, handType, (x_min - 30, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return self.lmList

    def get_fingers_status(self):
        fingersList = []
        for handLmList in self.lmList:
            fingers = []
            if len(handLmList) == 0:
                fingersList.append(fingers)
                continue

            # Thumb
            if self.handTypes[self.lmList.index(handLmList)] == "Right":
                if handLmList[self.tipIds[0]][1] < handLmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:  # Left hand
                if handLmList[self.tipIds[0]][1] > handLmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Fingers
            for id in range(1, 5):
                if handLmList[self.tipIds[id]][2] < handLmList[self.tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            fingersList.append(fingers)
        return fingersList

    def calculate_distance(self, p1, p2, img, handNo=0, draw=True):
        if len(self.lmList) == 0 or handNo >= len(self.lmList):
            return 0, img, []

        handLmList = self.lmList[handNo]
        x1, y1 = handLmList[p1][1], handLmList[p1][2]
        x2, y2 = handLmList[p2][1], handLmList[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    pTime = 0
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
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)
        cv2.imshow('Image', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

