from main import hand_detector
from PIL import Image
import cv2
import time
import pandas as pd
import numpy as np
from math import floor
from random import shuffle

DATAPATH = "./data"

LEFTCLICK  = 0
SCROLL = 6
MOVE       = 10
RIGHTCLICK = 20

GESTURES_ID = [
    LEFTCLICK,
    RIGHTCLICK,
    MOVE,
    SCROLL
]

GESTURES_TEXT = [
    "Left click",
    "Right click",
    "Open hand",
    "ASL 3"
]

def convert_landmarks_to_features(lmList):
    features = []
    origin = lmList[0]
    for _, x, y in lmList:
        features.append(x - origin[1])
        features.append(y - origin[2])
    return features

if __name__ == "__main__":
    output_file = open(DATAPATH + "/custom_points.csv", "a")
    cap = cv2.VideoCapture(0)
    detector = hand_detector(mode=True, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5)
    
    b = list(range(20))
    shuffle(b)
    for i in b:
        start = time.time()

        while floor(time.time() - start) < 5:
            err, img = cap.read()
            detector.find_hands(img)
            detector.find_position(img)

            cv2.putText(img, "Gesture: [{}]".format(GESTURES_TEXT[i%len(GESTURES_ID)]), (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.putText(img, "Get Ready: {}".format(floor(start - time.time() + 5)), (10, 80), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2)
            cv2.imshow("Collection", img)
            cv2.waitKey(1)

        # Camera flash effect
        cv2.imshow("Collection", np.ones_like(img)*255)
        cv2.waitKey(1)
        time.sleep(0.1)

        # Collect hand data
        err, img = cap.read()
        img = detector.find_hands(img)
        lmList = detector.find_position(img)
        features = convert_landmarks_to_features(lmList)
        
        for pt in features:
            output_file.write("{},".format(pt))
        output_file.write(str(GESTURES_ID[i%len(GESTURES_ID)]) + ",\n")

        # Sleep for a second
        cv2.imshow("Collection", img)
        cv2.waitKey(1)
        time.sleep(2)

            
        
