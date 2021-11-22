import numpy as np
import cv2
import mediapipe as mp
import time

class hand_detector():
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.max_hands,
                                        min_detection_confidence=self.detection_confidence,
                                        min_tracking_confidence=self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        hand_landmarks = self.results.multi_hand_landmarks
        if hand_landmarks:
            for handLms in hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0):
        lmList = []

        if self.results.multi_hand_landmarks:
            curr_hand = self.results.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(curr_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])

        return lmList

class Face_Detector():
    def __init__(self):
        self.parameter = True

class Bounding_Box():
    def __init__(self):
        self.parameter = True

class Right_Hand():
    def __init__(self):
        self.xloc = 0
        self.yloc = 0
        self.parameter = True

    def update(self, x, y, update_loc):
        if update_loc:
            self.xloc = x
            self.yloc = y

    def find_hand(self, img, handLm, draw=True):
        if handLm and draw:
            self.mpDraw.draw_landmarks(img, handLm, self.mpHands.HAND_CONNECTIONS)
        return img

# class Left_Hand():
#     def __init__(self):
#         self.parameter = True


if __name__ == '__main__':
    cTime = 0
    pTime = 0

    cap = cv2.VideoCapture(0)
    detector = hand_detector()


    while True:
        success, img = cap.read()

        img = detector.find_hands(img)
        lmList = detector.find_position(img)

        if lmList:
            features = []
            origin = lmList[0]
            for n, x, y in lmList:
                features.append(x - origin[1])
                features.append(y - origin[2])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
