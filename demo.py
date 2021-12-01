import cv2
import mediapipe as mp
import time
import math
from scipy import stats
import mouse
from eval import Evaluator
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
LEFTHOLD = 0
NOCLICK = 1
MIDDLECLICK = 2
LEFTCLICK = 3
RIGHTCLICK = 4

class hand_detector():
    def __init__(self, mode=False, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        max_num_hands = self.max_hands,
                                        min_detection_confidence = self.detection_confidence,
                                        min_tracking_confidence = self.tracking_confidence)
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

def get_mouse_position(lmlist):
    x, y = 0, 0
    for i in [0, 1, 5, 9, 13, 17]:
        x += lmlist[i][1]
        y += lmlist[i][2]
    
    return x / len(lmlist), y / len(lmlist)

if __name__ == '__main__':
    cTime = 0
    pTime = 0

    cap = cv2.VideoCapture(0)
    detector = hand_detector()

    px = 0
    py = 0
    cx = 0
    cy = 0

    # #attempt at averaging for smoothing
    # window_size = 5
    # positions = [[0, 0]] * window_size

    xMax = -math.inf
    yMax = -math.inf
    xMin = math.inf
    yMin = math.inf

    move = False

    xSens = 1
    ySens = 1

    timer = time.time()

    while (time.time() - timer < 10):
        success, img = cap.read()

        img = detector.find_hands(img)
        lmList = detector.find_position(img)

        if (lmList):
            xMax = max(xMax, lmList[0][1])
            yMax = max(yMax, lmList[0][2])
            xMin = min(xMin, lmList[0][1])
            yMin = min(yMin, lmList[0][2])
        
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
    xSens = 4*1920 / (xMax - xMin)
    ySens = 4*1080 / (yMax - yMin)
    print("Calibration done! Please wait for 3 seconds before system starts")
    print(xSens, ySens)

    timer = time.time()

    while (time.time() - timer < 2):
        continue

    print("System ready!")

    e = Evaluator("./data/model_87")
    gestures = [-1, -1, -1]


    mousePressed = None
    resetMiddleClick = True
    filter = [[0, 0]] * 3
    while True:
        success, img = cap.read()

        img = detector.find_hands(img)
        lmList = detector.find_position(img)

        if (lmList):            
            px = cx
            py = cy

            nx, ny = get_mouse_position(lmList)
            if move:
                filter.append([nx, ny])
                filter = filter[1:]
            else:
                filter = [[nx, ny]] * 3

            cx = sum([filter[i][0] for i in range(len(filter))]) / len(filter)
            cy = sum([filter[i][1] for i in range(len(filter))]) / len(filter)

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
            
            print(gesture, "{:.2f}%".format(confidence*100))

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

                if (move) and (abs(px - cx)**2 + abs(py - cy)**2)**0.5 > 0.05:
                    mouse.move(xSens*(px - cx), ySens*(cy - py),absolute=False)
                else:
                    move = True
        else:
            move = False

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)