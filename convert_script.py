from main import hand_detector
from PIL import Image
import cv2
import os
import pandas as pd
from math import floor

DATAPATH = "./data"

DATALABELS = {
    'Punch_VFR': 0,
    'Punch_VFL': 1,
    'One_VFR': 2,
    'One_VFL': 3,
    'Two_VFR': 4,
    'Two_VFL': 5,
    'Three_VFR': 6,
    'Three_VFL': 7,
    'Four_VFR': 8,
    'Four_VFL': 9,
    'Five_VFR': 10,
    'Five_VFL': 11,
    'Six_VFR': 12,
    'Six_VFL': 13,
    'Seven_VFR': 14,
    'Seven_VFL': 15,
    'Eight_VFR': 16,
    'Eight_VFL': 17,
    'Nine_VFR': 18,
    'Nine_VFL': 19,
    'Span_VFR': 20,
    'Span_VFL': 21,
    'Horiz_HBL': 22,
    'Horiz_HFL': 23,
    'Horiz_HBR': 24,
    'Horiz_HFR': 25,
}

INV_MAP = {v: k for k, v in DATALABELS.iteritems()}

def do_subject(dir):
    detector = hand_detector(mode=True, max_hands=1, detection_confidence=0.5, tracking_confidence=0.5)

    annot = pd.read_csv(dir + ".txt")
    annot = annot.drop(columns=['depth', 'Collab', 'XSign', 'TimeOut'])

    # out = open(dir + ".csv", "w")

    for row_i in range(annot.shape[0]):
        # Check the file exists
        png_path = dir + '/' + annot['rgb'][row_i].split('\\')[-1]
        if not os.path.isfile(png_path):
            continue
        
        for k in annot.loc[row_i].keys()[1:]:
            # Check if this images contains a class
            if annot.loc[row_i][k] == '[0 0 0 0]':
                continue
            
            bb = annot.loc[row_i][k].strip("[").strip("]").split(" ")
            bb = [int(floor(float(a))) for a in bb]
            
            img = cv2.imread(png_path)
            cv2.imshow("Image", img)
            cv2.waitKey(0)

            crop_img = img[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]

            cv2.imshow("Image", crop_img)
            cv2.waitKey(0)

            detector.find_hands(crop_img)
            cv2.imshow("Image", crop_img)
            cv2.waitKey(0)

            pts = detector.find_position(crop_img)
            if len(pts) != 21:
                print(pts)
                continue

            # origin = pts[0]
            # for pt in pts:
                # out.write("{x},{y},".format(x = pt[1] - origin[1], y = pt[2] - origin[2]))
            # out.write(str(DATALABELS[k]) + ",\n")
    # out.close()

if __name__ == "__main__":
    for subject in os.listdir(DATAPATH):
        dir = DATAPATH + "/" + subject + "/" + subject
        do_subject(dir)
        
        

        