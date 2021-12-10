import cv2
import mediapipe as mp
import time
import math
from scipy import stats
import mouse
from eval import Evaluator
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
LEFTHOLD = 0
NOCLICK = 1
MIDDLECLICK = 2
LEFTCLICK = 3
RIGHTCLICK = 4


class HandDetector:
    def __init__(self, mode=False, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        max_num_hands=self.max_hands,
                                        min_detection_confidence=self.detection_confidence,
                                        min_tracking_confidence=self.tracking_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        """
        Draw's hand landmarks from MediaPipe library on input image
        :param img: Display image for cv2 imshow
        :param draw:  Whether or not to draw the hands on input image
        :return: Input image with or without drawn landmarks
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        hand_landmarks = self.results.multi_hand_landmarks
        if hand_landmarks:
            for handLms in hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_number=0):
        """
        Returns landmark list positions of the detector
        :param img: Display image for cv2 imshow
        :param hand_number: Hand number of hand to extract landmarks from
        :return: landmark list of current hand
        """
        lmList = []

        if self.results.multi_hand_landmarks:
            curr_hand = self.results.multi_hand_landmarks[hand_number]

            for id, lm in enumerate(curr_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

        return lmList


class RightHand:
    def __init__(self, evaluator=1):
        # position estimation attributes
        self.px = 0
        self.py = 0
        self.cx = 0
        self.cy = 0
        self.xSens = 1
        self.ySens = 1
        self.filter = [[0., 0.]] * 3

        # gesture recognition attributes
        self.evaluator = evaluator

        # mouse action parameters
        self.mousePressed = None
        self.resetMiddleClick = True

        # fps display variables
        self.pTime = 0

    def display_fps(self, img):
        """
        Displays the fps of the video stream on input image
        :param img: display image for cv2 imshow
        :return: None
        """
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    def calibrate(self, hand_detector, show_fps=False):
        """
        Calibrates RightHand class x, y sensitivity (xSens, ySens)
        :param hand_detector: hand_detector for hand detection to obtain landmark list
        :param show_fps: show fps or not
        :return: None
        """
        xMax = -math.inf
        yMax = -math.inf
        xMin = math.inf
        yMin = math.inf

        timer = time.time()
        while time.time() - timer < 5:
            success, img = cap.read()
            img = hand_detector.find_hands(img)
            lmList = hand_detector.find_position(img)

            if lmList:
                xMax = max(xMax, lmList[0][1])
                yMax = max(yMax, lmList[0][2])
                xMin = min(xMin, lmList[0][1])
                yMin = min(yMin, lmList[0][2])

            if show_fps:
                self.display_fps(img)

            cv2.imshow("Image", img)
            cv2.waitKey(1)

        self.xSens = 4 * 1920 / (xMax - xMin)
        self.ySens = 8 * 1080 / (yMax - yMin)

        print("Calibration done! Please wait for 3 seconds before system starts")
        print(self.xSens, self.ySens)

        timer = time.time()
        while time.time() - timer < 2:
            continue

        print("System ready!")

    def get_mouse_position(self, lmlist):
        x, y = 0, 0
        for i in [0, 1, 5, 9, 13, 17]:
            x += lmlist[i][1]
            y += lmlist[i][2]

        return x / len(lmlist), y / len(lmlist)

    def update_position(self, lmList):
        """
        Updates RightHand class position given hand landmarks from detector
        :param lmList: detector landmark list
        :return: None
        """

        self.px, self.py = self.cx, self.cy

        nx, ny = self.get_mouse_position(lmList)
        self.filter.append([nx, ny])
        self.filter = self.filter[1:]

        self.cx = sum([self.filter[i][0] for i in range(len(self.filter))]) / len(self.filter)
        self.cy = sum([self.filter[i][1] for i in range(len(self.filter))]) / len(self.filter)

    def move_mouse(self):
        """
        Moves mouse on user interface
        :return: None
        """
        mouse.move(self.xSens * (self.px - self.cx), self.ySens * (self.cy - self.py), absolute=False)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    hand = RightHand()
    hand.calibrate(detector)

    move = False

    e = Evaluator("./data/model_87")
    gestures = [-1, -1, -1]

    mousePressed = None
    resetMiddleClick = True

    while True:
        success, img = cap.read()

        img = detector.find_hands(img)
        lmList = detector.find_position(img)

        if lmList:
            hand.update_position(lmList)
            px, py, cx, cy = hand.px, hand.py, hand.cx, hand.cy


            features = []
            origin = lmList[0]
            for n, x, y in lmList:
                features.append(x - origin[1])
                features.append(y - origin[2])

            gesture_id, confidence = e.eval(features)
            gestures = [gesture_id[0]] + gestures
            gestures = gestures[:-1]
            gesture, count = stats.mode(gestures)
            gesture = gesture[0]

            print(gesture, "{:.2f}%".format(confidence * 100))

            if count >= 3:
                if gesture == LEFTHOLD:
                    if mousePressed == None:
                        print("LEFTHOLD")
                        mouse.press(button=mouse.LEFT)
                        mousePressed = mouse.LEFT
                    else:
                        print("LEFT HELD")
                if gesture == LEFTCLICK:
                    if mousePressed == None:
                        print("LEFTCLICK")
                        mouse.click()
                        mousePressed = mouse.LEFT
                elif gesture == RIGHTCLICK:
                    if mousePressed == None:
                        print("RIGHTCLICK")
                        mouse.press(button=mouse.RIGHT)
                        mousePressed = mouse.RIGHT
                        resetMiddleClick = False
                elif gesture == MIDDLECLICK:
                    if mousePressed == None and resetMiddleClick:
                        print("MIDDLECLICK")
                        mouse.press(button=mouse.MIDDLE)
                        mousePressed = mouse.MIDDLE
                    if mousePressed == mouse.MIDDLE and resetMiddleClick:
                        mouse.release(button=mouse.MIDDLE)
                        resetMiddleClick = False
                elif gesture in [NOCLICK]:
                    if not mousePressed == None:
                        print("NOCLICK")
                        mouse.release(button=mousePressed)
                        resetMiddleClick = True
                        mousePressed = None

                if move and (abs(px - cx) ** 2 + abs(py - cy) ** 2) ** 0.5 > 0.05:
                    hand.move_mouse()
                else:
                    move = True

        else:
            move = False

        hand.display_fps(img)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
