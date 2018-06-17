import glob
import pathlib

import cmath
import dlib
import cv2
import imutils
import numpy as np
import shutil
import datetime
# OPEN CV
import os
import math
from imutils import face_utils
from imutils.face_utils import FaceAligner
from skimage.measure import compare_ssim as ssim
from skimage.color import rgb2gray
from sklearn.preprocessing import MinMaxScaler
import json
from skimage import filters
from skimage.color import rgb2gray
from skimage import data
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from skimage.filters import gaussian_filter
from sklearn import datasets
from skimage import exposure
from sklearn.model_selection import train_test_split
import tensorflow as tf
from mtcnn_facematch import detect_face
from skimage.transform import rotate
from PIL import Image

from skimage import exposure

import matplotlib.pyplot as plt

from mtcnn.mtcnn import MTCNN
from time import sleep
import argparse

from skimage import io
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from sklearn import neighbors

from dataStorage import DataStorage
from skimage import exposure
from skimage import feature
from skimage import img_as_ubyte

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import BallTree
from imutils.face_utils.helpers import FACIAL_LANDMARKS_IDXS
from imutils.face_utils.helpers import shape_to_np
import numpy as np
import cv2
from skimage import img_as_ubyte

###STAŁE
predictor_path = "landmark/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Core/landmark/vgg_face_caffe/vgg_face_caffe/VGG_FACE_deploy.prototxt
# net = cv2.dnn.readNetFromCaffe("landmark/PAM_frontal_AlexNet/PAM_frontal_deploy.prototxt.txt", "landmark/PAM_frontal_AlexNet/snap__iter_100000.caffemodel")


faceFolderPath = "Pozytywne/*"
badFaceFolderPath = "Negatywne/"
positiveLister = glob.glob(faceFolderPath)
# HaarCascade prepare data
haarFaceCascade = cv2.CascadeClassifier('HaarCascadeConfigs/haarcascade_frontalface_default[0]ml')
lbpCascade = cv2.CascadeClassifier('HaarCascadeConfigs/lbpcascade_frontalface_improved[0]ml')
harrProfileFaceCascade = cv2.CascadeClassifier('HaarCascadeConfigs/haarcascade_profileface.xml')
chinHeightROI = 0.23
confidenceOfDetection = 0.5
imageSizeToResize = 150

haarGoodPath = "WynikiAnalizy\\Haar Cascade\\Dobre\\"
haarBadPath = "WynikiAnalizy\\Haar Cascade\\Zle\\"
lbpGoodPath = "WynikiAnalizy\\Haar Cascade\\Dobre\\"
lbpBadPath = "WynikiAnalizy\\LBP\\Zle\\"
dlibGoodPath = "WynikiAnalizy\\Dlib\\Dobre\\"
dlibBadPath = "WynikiAnalizy\\Dlib\\Zle\\"

personDefPath = "WynikiAnalizy\\ProbkiBadawcze\\"
researchDefPath = "WynikiAnalizy\\"
### ZMIENNE
printDetails = True

goodResult = 0
badResult = 0

goodDeepLearning = 0
badDeepLearning = 0
### WYGENEROWANE

analysedData = []
positiveData = 0
negativeData = 0
positiveDataTest = 0
negativeDataTest = 0

sess = tf.Session()
# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')

### Sprawdzenie czy istnieje plik do logów
getTime = str(datetime.datetime.now().ctime())
if not (pathlib.Path("LogFile_Etap2.txt").is_file()):
    # os.mknod("/LogFile.txt",0)
    file = open("LogFile_Etap2.txt", 'a')
    file.writelines(
        "\n##################################################################### " + "\nTest : " + getTime + "\n\n")
else:
    file = open("LogFile_Etap2.txt", 'a')
    file.writelines(
        "\n##################################################################### " + "\nTest : " + getTime + "\n\n")


def dlibFaceDetector(inputFilePath, goodPath, badPath):
    if printDetails:
        file.writelines(
            getTime + "\t" + "Histogram of Oriented Gradients: (neighbours:\t")
    global badResult, goodResult

    inputFile = cv2.imread(inputFilePath)
    inputFile = imutils.resize(inputFile, 1100)
    # height, width = inputFile.shape[:2]
    # if (width < 600 or height < 600):
    #     inputFile = imutils.resize(inputFile, 1000)
    #     height, width = inputFile.shape[:2]
    # elif (width > 2000 or height > 2000):
    #     inputFile = imutils.resize(inputFile, int(width * 0.80), int(height * 0.80))
    #     height, width = inputFile.shape[:2]

    #
    # ( Width [0], Height [1]
    # inputFile = imutils.resize(inputFile, 500)
    greyImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    height, width = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")

    rects = detector(greyImage, 1)
    x = 0
    y = 0
    w = 0
    h = 0

    if len(rects) != 1:
        # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
    else:
        goodResult += 1
        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(greyImage, rect)
            # nomnom = predictor.full_object_detection(shape)
            shape = face_utils.shape_to_np(shape)

            # Pokazanie że wykrywa twarz - można pominąć
            # cv2.rectangle(inputFile, (x, y), (x + w, y + int(h+(h*0.2))), (255, 0, 0), 2)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box

            # udowodnienie że twarz wykrywa
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        if x < 0:
            x = 0
        elif x > width:
            x = width - 1

        if (y < 0):
            y = 0
        elif y > height:
            y = height - 1

        if w < 0:
            w = 0
        elif w > width:
            w = width - 1

        if (h < 0):
            h = 0
        elif h > height:
            h = height - 1

        # cv2.rectangle(inputFile, (x, y), (x + w, y + h), (0, 255, 0), 2)

        smart_h = int(h * chinHeightROI)
        roi_color = inputFile[y:y + h, x:x + w]

        roi_gray = greyImage[y:y + height, x:x + w]
        index = 0

        fp0 = [shape[0][0], shape[0][1]]
        fp1 = [shape[1][0], shape[1][1]]
        fp2 = [shape[2][0], shape[2][1]]
        fp3 = [shape[3][0], shape[3][1]]
        fp4 = [shape[4][0], shape[4][1]]
        fp5 = [shape[5][0], shape[5][1]]
        fp6 = [shape[6][0], shape[6][1]]
        fp7 = [shape[7][0], shape[7][1]]
        fp8 = [shape[8][0], shape[8][1]]
        fp9 = [shape[9][0], shape[9][1]]
        fp10 = [shape[10][0], shape[10][1]]
        fp11 = [shape[11][0], shape[11][1]]
        fp12 = [shape[12][0], shape[12][1]]
        fp13 = [shape[13][0], shape[13][1]]
        fp14 = [shape[14][0], shape[14][1]]
        fp15 = [shape[15][0], shape[15][1]]
        fp16 = [shape[16][0], shape[16][1]]
        fp17 = [shape[17][0], shape[17][1]]
        fp18 = [shape[18][0], shape[18][1]]
        fp19 = [shape[19][0], shape[19][1]]
        fp20 = [shape[20][0], shape[20][1]]
        fp21 = [shape[21][0], shape[21][1]]
        fp22 = [shape[22][0], shape[22][1]]
        fp23 = [shape[23][0], shape[23][1]]
        fp24 = [shape[24][0], shape[24][1]]
        fp25 = [shape[25][0], shape[25][1]]
        fp26 = [shape[26][0], shape[26][1]]
        fp27 = [shape[27][0], shape[27][1]]
        fp28 = [shape[28][0], shape[28][1]]
        fp29 = [shape[29][0], shape[29][1]]  # ponad czubkiemn nosa
        fp30 = [shape[30][0], shape[30][1]]  # czubek nosa
        fp31 = [shape[31][0], shape[31][1]]
        fp32 = [shape[32][0], shape[32][1]]
        fp33 = [shape[33][0], shape[33][1]]
        fp34 = [shape[34][0], shape[34][1]]
        fp35 = [shape[35][0], shape[35][1]]
        fp36 = [shape[36][0], shape[36][1]]  # lewy zewnętrzny kącik oka
        fp37 = [shape[37][0], shape[37][1]]
        fp38 = [shape[38][0], shape[38][1]]
        fp39 = [shape[39][0], shape[39][1]]  # lewy wewnetrzny kącik oka
        fp40 = [shape[40][0], shape[40][1]]  # dolna zrenica wewnetrzna
        fp41 = [shape[41][0], shape[41][1]]  # dolna zrenica zewnetrzna
        fp42 = [shape[42][0], shape[42][1]]  # prawy wewnetrzny kacik oka
        fp43 = [shape[43][0], shape[43][1]]
        fp44 = [shape[44][0], shape[44][1]]
        fp45 = [shape[45][0], shape[45][1]]  # prawy zewnetrzny kacik oka
        fp46 = [shape[46][0], shape[46][1]]  # zewnetrzna prawwa zrenica
        fp47 = [shape[47][0], shape[47][1]]  # wewnetrzna prawa zrenica
        fp48 = [shape[48][0], shape[48][1]]  # lewy kącik ust
        fp49 = [shape[49][0], shape[49][1]]  # lewa warga zewnątrz
        fp50 = [shape[50][0], shape[50][1]]
        fp51 = [shape[51][0], shape[51][1]]  # środek górnej wargi
        fp52 = [shape[52][0], shape[52][1]]
        fp53 = [shape[53][0], shape[53][1]]  # prawa warga zewnątrz góra
        fp54 = [shape[54][0], shape[54][1]]  # prawy kącik ust
        fp55 = [shape[55][0], shape[55][1]]  # prawa warga zewnątrz dół
        fp56 = [shape[56][0], shape[56][1]]  # środek dolnej wargi
        fp57 = [shape[57][0], shape[57][1]]
        fp58 = [shape[58][0], shape[58][1]]
        fp59 = [shape[59][0], shape[59][1]]  # lewa warga dół
        fp60 = [shape[60][0], shape[60][1]]
        fp61 = [shape[61][0], shape[61][1]]
        fp62 = [shape[62][0], shape[62][1]]
        fp63 = [shape[63][0], shape[63][1]]
        fp64 = [shape[64][0], shape[64][1]]
        fp65 = [shape[65][0], shape[65][1]]
        fp66 = [shape[66][0], shape[66][1]]
        fp67 = [shape[67][0], shape[67][1]]

        # for (x, y) in shape:
        #     # xp1 = shape[0,0]
        #     cv2.circle(inputFile, (x, y), 1, (0, 0, 255), 5)
        #     cv2.imshow("image", inputFile)
        #     cv2.waitKey(0)
        # print (index)
        # # print(str(x))
        # # print(str(y))
        # index+=1
        # cv2.imshow("image", inputFile)
        # cv2.waitKey(0)

        #     obliczanie czy twarz jest przechylonwa wzdłuż jaw
        fa = FaceAligner(predictor, desiredFaceWidth=width, desiredFaceHeight=height, desiredLeftEye=(0.39, 0.39))
        faceOrig = imutils.resize(inputFile[y:y + h, x:x + w], width=256)
        xCenter = (int)((w / 2) + x)
        yCenter = (int)((h / 2) + y)
        lSideLength = math.sqrt(math.pow(fp48[0] - fp4[0], 2) + (math.pow(fp48[1] - fp4[1], 2)))
        rSideLength = math.sqrt(math.pow(fp54[0] - fp12[0], 2) + (math.pow(fp54[1] - fp12[1], 2)))
        if (((lSideLength * 0.666) > rSideLength) or ((rSideLength * 0.666) > lSideLength)):
            greyImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
            # for (x, y) in shape:
            #     cv2.circle(greyImage, (x, y), 1, (0, 0, 255), 5)

            # cv2.circle(greyImage, (xCenter, yCenter), 1, (200, 255, 23), 5)
            # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, greyImage)
        else:
            greyImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
            faceAligned = fa.align(inputFile, greyImage, rect)
            grayImageAligned = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("Original", grayImageAligned)

            rects = detector(grayImageAligned, 1)

            # for (x, y) in shape:
            #     cv2.circle(inputFile, (x, y), 1, (0, 0, 255), 5)
            # cv2.circle(inputFile, (xCenter, yCenter), 1, (200, 255, 23), 5)
            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)
            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, faceAligned)
            # cv2.imshow("Original", faceOrig)
            # cv2.imshow("Aligned", faceAligned)
            # cv2.waitKey(0)
            x = 0
            y = 0
            w = 0
            h = 0

            if len(rects) != 1:
                # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
                badResult += 1
            else:
                goodResult += 1
                for (i, rect) in enumerate(rects):
                    # determine the facial landmarks for the face region, then
                    # convert the facial landmark (x, y)-coordinates to a NumPy
                    # array
                    shape = predictor(grayImageAligned, rect)
                    # nomnom = predictor.full_object_detection(shape)
                    shape = face_utils.shape_to_np(shape)

                # Pokazanie że wykrywa twarz - można pominąć
                # cv2.rectangle(inputFile, (x, y), (x + w, y + int(h+(h*0.2))), (255, 0, 0), 2)
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box

                # udowodnienie że twarz wykrywa
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                if x < 0:
                    x = 0
                elif x > width:
                    x = width - 1

                if (y < 0):
                    y = 0
                elif y > height:
                    y = height - 1

                if w < 0:
                    w = 0
                elif w > width:
                    w = width - 1

                if (h < 0):
                    h = 0
                elif h > height:
                    h = height - 1
            xCenter = (int)((w / 2) + x)
            yCenter = (int)((h / 2) + y)
            # cv2.rectangle(faceAligned, (x, y), (x + w, y + h), (0, 255, 0), 2)
            counter = 0
            # for (x, y) in shape:
            #     cv2.circle(faceAligned, (x, y), 1, (0, 0, 255), 5)
            #     counter += 1
            #     print(counter)
            #     cv2.imshow("Aligned", faceAligned)
            #     cv2.waitKey(0)

            # cv2.circle(faceAligned, (xCenter, yCenter), 1, (200, 255, 23), 5)

            # sprawdzenie po powtórnym przypisaniu czy twarz na pewno ma rozmiar twarzy czy też się nie zgubiła w procesie powtórenego przypisania punktów twarzy
            # brane jest połowa długości rect i porównywana z totalną szerokością twarzy
            fp4 = [shape[4][0], shape[4][1]]
            fp12 = [shape[12][0], shape[12][1]]

            totalFaceLength = math.sqrt(math.pow(fp12[0] - fp4[0], 2) + (math.pow(fp12[1] - fp4[1], 2)))
            faceRectSize = 0.5 * (math.sqrt(math.pow(x - (x + w), 2) + (math.pow(y - y, 2))))
            if (totalFaceLength > faceRectSize):
                analysedData.append(DataStorage(inputFilePath, 0, inputFile, faceAligned, shape, False))
                print("analysedData: " + str(analysedData.__len__()))
            else:
                print("Zdjecie odrzucone")
                cv2.imshow("Aligned", faceAligned)
                cv2.waitKey(0)

            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, faceAligned)

            # cv2.imshow("Aligned", faceAligned)
            # cv2.waitKey(0)


def checkROIOfImage(bounding_boxes, inputFile, width, height, markData):
    rect = None
    x = None
    y = None
    hPoint = None
    wPoint = None
    if (len(bounding_boxes) == 0):
        # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        # cv2.imshow("image", inputFile)
        # cv2.waitKey(0)
        # badResult += 1
        # cv2.imshow("odrzucony na rozpoznaniu",inputFile)
        # cv2.waitKey(0)
        return False, rect, x, y, hPoint, wPoint, inputFile

    elif (len(bounding_boxes) == 1):

        if (bounding_boxes[0][4] > 0.95):
            # print("highest: " + str(bounding_boxes[0][4]))

            x, y, wPoint, hPoint = int(bounding_boxes[0][0]), int(bounding_boxes[0][1]), int(bounding_boxes[0][2]), int(
                bounding_boxes[0][3])

            if x < 0:
                x = 0
            elif x > width:
                x = width - 1

            if (y < 0):
                y = 0
            elif y > height:
                y = height - 1

            if wPoint < 0:
                wPoint = 0
            elif wPoint > width:
                wPoint = width - 1

            if (hPoint < 0):
                hPoint = 0
            elif hPoint > height:
                hPoint = height - 1

            if (markData):
                cv2.rectangle(inputFile, (x, y),
                              (wPoint, hPoint), (0, 255, 0), 2)
            # goodResult += 1
            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)
            rect = dlib.rectangle(x, y, wPoint, hPoint)
            return True, rect, x, y, hPoint, wPoint, inputFile

        else:
            # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
            return False, rect, x, y, hPoint, wPoint, inputFile

            # cv2.imshow("odrzucony na rozpoznaniu",inputFile)
            # cv2.waitKey(0)

    elif (len(bounding_boxes) != 1):

        highest = bounding_boxes[0]
        for i in range(1, len(bounding_boxes), 1):
            # print("highest: " + str(highest[0]))

            if (bounding_boxes[i][4] > highest[4]):
                highest = bounding_boxes[i]

        # print("highest: " + str(highest[0]))

        if (highest[4] > 0.95):
            for a in range(0, 4, 1):
                highest[a] = int(highest[a])

            x, y, wPoint, hPoint = int(highest[0]), int(highest[1]), int(highest[2]), int(highest[3])

            if x < 0:
                x = 0
            elif x > width:
                x = width - 1

            if (y < 0):
                y = 0
            elif y > height:
                y = height - 1

            if wPoint < 0:
                wPoint = 0
            elif wPoint > width:
                wPoint = width - 1

            if (hPoint < 0):
                hPoint = 0
            elif hPoint > height:
                hPoint = height - 1

            rect = dlib.rectangle(x, y, wPoint, hPoint)
            if (markData):
                cv2.rectangle(inputFile, (x, y),
                              (wPoint, hPoint), (0, 255, 0), 2)
            # goodResult += 1
            return True, rect, x, y, hPoint, wPoint, inputFile
            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)
    else:
        # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        # badResult += 1
        return False, rect, x, y, hPoint, wPoint, inputFile


def alignRect(x, y, w, h, width, height):
    if x < 0:
        x = 0
    elif x > width:
        x = width - 1

    if (y < 0):
        y = 0
    elif y > height:
        y = height - 1

    if w < 0:
        w = 0
    elif w > width:
        w = width - 1

    if (h < 0):
        h = 0
    elif h > height:
        h = height - 1
    return x, y, w, h


def alignMyFace(image, gray, rect, desiredLeftEye, desiredFaceHeight, desiredFaceWidth, shape):
    # extract the left and right eye (x, y)-coordinates
    (lStart, lEnd) = FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = FACIAL_LANDMARKS_IDXS["right_eye"]
    leftEyePts = shape[lStart:lEnd]
    rightEyePts = shape[rStart:rEnd]

    # compute the center of mass for each eye
    leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
    rightEyeCenter = rightEyePts.mean(axis=0).astype("int")
    fp39 = [shape[39][0], shape[39][1]]  # lewy wewnetrzny kącik oka
    fp42 = [shape[42][0], shape[42][1]]  # prawy wewnetrzny kacik oka
    # compute the angle between the eye centroids
    dY = rightEyeCenter[1] - leftEyeCenter[1]
    dX = rightEyeCenter[0] - leftEyeCenter[0]

    dYY = fp39[1] - fp42[1]
    dXX = fp39[0] - fp42[0]
    # for (xx, yy) in shape:
    #     cv2.circle(image, (xx, yy), 1, (0, 0, 255), 5)
    angle = np.degrees(np.arctan2(dY, dX)) - 180

    # angle = np.degrees(math.atan2(fp42[1] - fp39[1], fp39[0] - fp42[0]) )  * (180.0 / math.pi)
    # angle = np.degrees(np.arctan2(dYY, dXX))

    # eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
    #               (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
    eyesCenter2 = ((fp39[0] + fp42[0]) / 2, (fp39[1] + fp42[1]) / 2)
    # cv2.imshow("przed rotacja", image)
    # cv2.waitKey(0)
    simage = rotate(image, angle, resize=False, center=eyesCenter2, order=1, mode='reflect', cval=0, clip=True,
                    preserve_range=False)  # reflect symmetric
    cv_image = img_as_ubyte(simage)
    # for i in range (0,-359,-10):
    #     simage = rotate(image, i, resize=False, center=eyesCenter2, order=1, mode='wrap', cval=0, clip=True,
    #                     preserve_range=False)
    #     cv_image = img_as_ubyte(simage)

    # vis = np.concatenate((image, cv_image), axis=1)
    # cv2.imshow("kat zmiany " + str(angle), vis)
    print("kat zmiany " + str(angle))
    cv2.waitKey(0)
    # reflect symmetric
    return cv_image
    # convert the landmark (x, y)-coordinates to a NumPy array
    # shape = predictor(gray, rect)
    # shape = shape_to_np(shape)


def getFace(inputFilePath, threshold, factor, goodPath, badPath):
    global goodResult, badResult
    inputFile = cv2.imread(inputFilePath)
    height, width = inputFile.shape[:2]
    # if (width < 600 or height < 600):
    #     inputFile = imutils.resize(inputFile, 800)
    # height, width = inputFile.shape[:2]
    if (width > 2000 or height > 2000):
        inputFile = imutils.resize(inputFile, 1500)
        height, width = inputFile.shape[:2]
    # else:
    #     inputFile = imutils.resize(inputFile, width=1100)

    height, width = inputFile.shape[:2]
    bounding_boxes, _ = detect_face.detect_face(inputFile, int(width * 0.2), pnet, rnet, onet, threshold, factor)

    isGood, rect, x, y, hPoint, wPoint, inputFile = checkROIOfImage(bounding_boxes, inputFile, width, height, False)
    if ((isGood == False) or (rect == None)):
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
        return None
    #####################################################################################

    greyImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    shape = predictor(greyImage, rect)
    shape = face_utils.shape_to_np(shape)

    fp0 = [shape[0][0], shape[0][1]]
    fp1 = [shape[1][0], shape[1][1]]
    fp2 = [shape[2][0], shape[2][1]]
    fp3 = [shape[3][0], shape[3][1]]
    fp4 = [shape[4][0], shape[4][1]]
    fp5 = [shape[5][0], shape[5][1]]
    fp6 = [shape[6][0], shape[6][1]]
    fp7 = [shape[7][0], shape[7][1]]
    fp8 = [shape[8][0], shape[8][1]]
    fp9 = [shape[9][0], shape[9][1]]
    fp10 = [shape[10][0], shape[10][1]]
    fp11 = [shape[11][0], shape[11][1]]
    fp12 = [shape[12][0], shape[12][1]]
    fp13 = [shape[13][0], shape[13][1]]
    fp14 = [shape[14][0], shape[14][1]]
    fp15 = [shape[15][0], shape[15][1]]
    fp16 = [shape[16][0], shape[16][1]]
    fp17 = [shape[17][0], shape[17][1]]
    fp18 = [shape[18][0], shape[18][1]]
    fp19 = [shape[19][0], shape[19][1]]
    fp20 = [shape[20][0], shape[20][1]]
    fp21 = [shape[21][0], shape[21][1]]
    fp22 = [shape[22][0], shape[22][1]]
    fp23 = [shape[23][0], shape[23][1]]
    fp24 = [shape[24][0], shape[24][1]]
    fp25 = [shape[25][0], shape[25][1]]
    fp26 = [shape[26][0], shape[26][1]]
    fp27 = [shape[27][0], shape[27][1]]
    fp28 = [shape[28][0], shape[28][1]]
    fp29 = [shape[29][0], shape[29][1]]  # ponad czubkiemn nosa
    fp30 = [shape[30][0], shape[30][1]]  # czubek nosa
    fp31 = [shape[31][0], shape[31][1]]
    fp32 = [shape[32][0], shape[32][1]]
    fp33 = [shape[33][0], shape[33][1]]
    fp34 = [shape[34][0], shape[34][1]]
    fp35 = [shape[35][0], shape[35][1]]
    fp36 = [shape[36][0], shape[36][1]]  # lewy zewnętrzny kącik oka
    fp37 = [shape[37][0], shape[37][1]]
    fp38 = [shape[38][0], shape[38][1]]
    fp39 = [shape[39][0], shape[39][1]]  # lewy wewnetrzny kącik oka
    fp40 = [shape[40][0], shape[40][1]]  # dolna zrenica wewnetrzna
    fp41 = [shape[41][0], shape[41][1]]  # dolna zrenica zewnetrzna
    fp42 = [shape[42][0], shape[42][1]]  # prawy wewnetrzny kacik oka
    fp43 = [shape[43][0], shape[43][1]]
    fp44 = [shape[44][0], shape[44][1]]
    fp45 = [shape[45][0], shape[45][1]]  # prawy zewnetrzny kacik oka
    fp46 = [shape[46][0], shape[46][1]]  # zewnetrzna prawwa zrenica
    fp47 = [shape[47][0], shape[47][1]]  # wewnetrzna prawa zrenica
    fp48 = [shape[48][0], shape[48][1]]  # lewy kącik ust
    fp49 = [shape[49][0], shape[49][1]]  # lewa warga zewnątrz
    fp50 = [shape[50][0], shape[50][1]]
    fp51 = [shape[51][0], shape[51][1]]  # środek górnej wargi
    fp52 = [shape[52][0], shape[52][1]]
    fp53 = [shape[53][0], shape[53][1]]  # prawa warga zewnątrz góra
    fp54 = [shape[54][0], shape[54][1]]  # prawy kącik ust
    fp55 = [shape[55][0], shape[55][1]]  # prawa warga zewnątrz dół
    fp56 = [shape[56][0], shape[56][1]]  # środek dolnej wargi
    fp57 = [shape[57][0], shape[57][1]]
    fp58 = [shape[58][0], shape[58][1]]
    fp59 = [shape[59][0], shape[59][1]]  # lewa warga dół
    fp60 = [shape[60][0], shape[60][1]]
    fp61 = [shape[61][0], shape[61][1]]
    fp62 = [shape[62][0], shape[62][1]]
    fp63 = [shape[63][0], shape[63][1]]
    fp64 = [shape[64][0], shape[64][1]]
    fp65 = [shape[65][0], shape[65][1]]
    fp66 = [shape[66][0], shape[66][1]]
    fp67 = [shape[67][0], shape[67][1]]

    # fa = FaceAligner(predictor, desiredFaceWidth=int(width * 1.5), desiredFaceHeight=int(height * 1.5),
    #                  desiredLeftEye=(0.40, 0.40))  # 36

    sizeMouthFactor = 0.4
    sizeEyeFactor = 0.4  # 0.5
    # xCenter = (int)((w / 2) + x)
    # yCenter = (int)((h / 2) + y)
    lSideNoseLength = math.sqrt(math.pow(fp48[0] - fp30[0], 2) + (math.pow(fp48[1] - fp30[1], 2)))
    rSideNoseLength = math.sqrt(math.pow(fp54[0] - fp30[0], 2) + (math.pow(fp54[1] - fp30[1], 2)))

    lSideEyeLength = math.sqrt(math.pow(fp36[0] - fp0[0], 2) + (math.pow(fp36[1] - fp0[1], 2)))
    rSideEyeLength = math.sqrt(math.pow(fp45[0] - fp16[0], 2) + (math.pow(fp45[1] - fp16[1], 2)))
    if ((lSideNoseLength * sizeMouthFactor) > rSideNoseLength) or (
            (rSideNoseLength * sizeMouthFactor) > lSideNoseLength) or (
            (lSideEyeLength * sizeEyeFactor) > rSideEyeLength) or (
            (
                    rSideEyeLength * sizeEyeFactor) > lSideEyeLength):  # todo : dodać korektę na zmianę odległości kącików ust na coś innego
        # greyImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
        for (x, y) in shape:
            cv2.circle(inputFile, (x, y), 1, (0, 0, 255), 5)
        # cv2.imshow("test", inputFile)
        # cv2.waitKey(0)
        # cv2.circle(greyImage, (xCenter, yCenter), 1, (200, 255, 23), 5)
        # cv2.imshow("odrzucony na odchyleniach",inputFile)
        # cv2.waitKey(0)
        badResult += 1
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        return None

    # for (xx, yy) in shape:
    #     cv2.circle(inputFile, (xx, yy), 1, (0, 0, 255), 5)
    # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)
    # cv2.imshow("test", inputFile)
    # cv2.waitKey(0)

    ################ Przycinanie pierwszy raz###########################################
    xn = x - int((wPoint - x) * 0.4)
    yn = y - int((hPoint - y) * 0.4)
    wn = int((wPoint - x) * 1.8)
    hn = int((hPoint - y) * 1.8)

    xn, yn, wn, hn = alignRect(xn, yn, wn, hn, width, height)
    inputFile = inputFile[yn:yn + hn, xn:xn + wn]

    # cv2.imshow("cut image ", inputFile)
    # cv2.waitKey(0)
    height, width = inputFile.shape[:2]
    bounding_boxes, _ = detect_face.detect_face(inputFile, int(width * 0.2), pnet, rnet, onet, threshold, factor)

    isGood, rect, x, y, hPoint, wPoint, inputFile = checkROIOfImage(bounding_boxes, inputFile, width, height, False)
    greyImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    if ((isGood == False) or (rect == None)):
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
        return None

    shape = predictor(greyImage, rect)
    shape = face_utils.shape_to_np(shape)

    faceAligned = alignMyFace(inputFile, greyImage, rect, (0.38, 0.38), height, width, shape)
    # inputFile = alignMyFace(inputFile, greyImage, rect, width, height
    # cv2.imshow("after align ", faceAligned)
    # cv2.waitKey(0)
    height, width = faceAligned.shape[:2]
    bounding_boxes, _ = detect_face.detect_face(faceAligned, int(width * 0.2), pnet, rnet, onet, threshold, factor)

    isGood, rect, x, y, hPoint, wPoint, faceAligned = checkROIOfImage(bounding_boxes, faceAligned, width, height, False)
    if ((isGood == False) or (rect == None)):
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
        return None

    # cv2.imshow("aliged image again ", faceAligned)
    # cv2.waitKey(0)
    #####################################################################################
    ################ Przycinanie drugi raz###########################################
    # faceAligned
    xn = x - int((wPoint - x) * 0.2)
    yn = y - int((hPoint - y) * 0.2)
    wn = int((wPoint - x) * 1.4)
    hn = int((hPoint - y) * 1.4)
    xn, yn, wn, hn = alignRect(xn, yn, wn, hn, width, height)
    faceAlignedAndCropped = faceAligned[yn:yn + hn, xn:xn + wn]
    bounding_boxes, _ = detect_face.detect_face(faceAlignedAndCropped, int(width * 0.2), pnet, rnet, onet, threshold,
                                                factor)
    isGood, rect, x, y, hPoint, wPoint, faceAlignedAndCropped = checkROIOfImage(bounding_boxes, faceAlignedAndCropped,
                                                                                width, height, False)
    if ((isGood == False) or (rect == None)):
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
        return None
    greyImage = cv2.cvtColor(faceAlignedAndCropped, cv2.COLOR_BGR2GRAY)

    # shape = predictor(greyImage, rect)
    # shape = face_utils.shape_to_np(shape)
    # for (xx, yy) in shape:
    #     cv2.circle(faceAlignedAndCropped, (xx, yy), 1, (0, 0, 255), 5)
    # cv2.imshow("aliged and cropped face ", faceAlignedAndCropped)
    # cv2.waitKey(0)
    ################ normalizowanie do pożądanej wartości ###########################################

    faceResidedAlignedCropped = imutils.resize(faceAlignedAndCropped, width=1100)
    height, width = faceResidedAlignedCropped.shape[:2]
    bounding_boxes, _ = detect_face.detect_face(faceResidedAlignedCropped, int(width * 0.2),
                                                pnet, rnet, onet, threshold, factor)
    isGood, rect, x, y, hPoint, wPoint, faceResidedAlignedCropped = checkROIOfImage(bounding_boxes,
                                                                                    faceResidedAlignedCropped,
                                                                                    width, height, True)

    if ((isGood == False) or (rect == None)):
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
        return None

    greyImage = cv2.cvtColor(faceResidedAlignedCropped, cv2.COLOR_BGR2GRAY)
    shape = predictor(greyImage, rect)
    shape = face_utils.shape_to_np(shape)
    #
    #
    #
    #
    #
    #
    #
    #
    fp4 = [shape[4][0], shape[4][1]]
    fp12 = [shape[12][0], shape[12][1]]
    totalFaceWidth = math.sqrt(math.pow(fp12[0] - fp4[0], 2) + (math.pow(fp12[1] - fp4[1],
                                                                         2)))  # TODO : dodać bądź zmienić warunek z szerokości twarzy na szerokość oczu
    rectWidth = (math.sqrt(math.pow(x - wPoint, 2) + (math.pow(y - y, 2))))
    rectHeight = (math.sqrt(math.pow(y - hPoint, 2) + (math.pow(x - x, 2))))
    halfFaceRectWidth = 0.5 * rectWidth

    # KWADRATY KONTROLNE !!!
    xCenter = (int)((rectWidth / 2) + x)
    yCenter = (int)((rectHeight / 2) + y)
    cv2.circle(faceResidedAlignedCropped, (xCenter, yCenter), 1, (200, 255, 23), 5)

    # r1X,r1Y,r1W,r1H
    r1X = x
    r1Y = y
    r1W = int(0.29 * rectWidth)
    r1H = yCenter
    cv2.rectangle(faceResidedAlignedCropped, (r1X, r1Y),
                  (x + r1W, r1H), (255, 255, 0), 2)

    r2X = wPoint - int(0.295 * rectWidth)
    r2Y = y
    r2W = int(0.29 * rectWidth)
    r2H = yCenter
    # cv2.imshow("aaaa",faceAligned)
    # cv2.waitKey(0)
    cv2.rectangle(faceResidedAlignedCropped, (r2X, r2Y),
                  (wPoint, r2H), (0, 255, 200), 2)

    fp36 = [shape[36][0], shape[36][1]]  # lewy zewnętrzny kącik oka
    fp45 = [shape[45][0], shape[45][1]]  # prawy zewnetrzny kacik oka

    if ((fp36[0] > (x + r1W)) or (fp36[0] < x) or (fp36[1] > yCenter) or (fp36[1] < y)):
        lsideFailure = True
    else:
        lsideFailure = False

    if ((fp45[0] > (wPoint)) or (fp45[0] < r2X) or (fp45[1] > yCenter) or (fp45[1] < y)):
        rsideFailure = True
    else:
        rsideFailure = False

    if (totalFaceWidth > halfFaceRectWidth and lsideFailure == False and rsideFailure == False):
        analysedData.append(DataStorage(inputFilePath, 0, inputFile, faceResidedAlignedCropped, shape, False))
        print("analysedData passed! : " + str(analysedData.__len__()))
        # for (x, y) in shape:
        #     cv2.circle(faceResidedAlignedCropped, (x, y), 1, (0, 0, 255), 5)
        cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, faceResidedAlignedCropped)
        goodResult += 1
        return None

    else:
        # cv2.imshow("odrzucony na ostatniej walidacji",faceAligned)
        # cv2.waitKey(0)
        for (x, y) in shape:
            cv2.circle(faceResidedAlignedCropped, (x, y), 1, (0, 0, 255), 5)
        cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, faceResidedAlignedCropped)
        # cv2.imshow("Aligned", faceAligned)
        # cv2.waitKey(0)
        badResult += 1
        return None


def researchOrderer(alghoritmName, mode, values, clear):
    global printDetails
    global goodResult, badResult
    falsePositive = 0
    truePositive = 0
    falseNegative = 0
    trueNegative = 0

    global positiveLister
    global getXTime
    getTimeFolderPersons = datetime.datetime.now()
    getXTime = str(getTimeFolderPersons.strftime("%Y-%m-%d - %H-%M-%S"))
    if (alghoritmName == "MTCNN"):
        if (mode == "SICK"):
            file.writelines("Positive\t")
            print("MTCNN: Sick People")

            pathCore = researchDefPath + "MTCNN\\" + getXTime + " MTCNN " + "\\"
            pathCore = pathCore.replace(":", " ")

            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            os.mkdir(pathGood)
            os.mkdir(pathBad)
            counter = 0
            for image in positiveLister:
                print(image)
                print("Iteracja: " + str(counter))
                counter += 1
                # dlibFaceDetector(image, pathGood, pathBad)
                getFace(image, values[0], values[1], pathGood, pathBad)
                if printDetails:
                    printDetails = False
            printDetails = True
            file.writelines("Results:\t")
            file.writelines("Good:\t" + str(goodResult) + '\t')
            file.writelines("Bad:\t" + str(badResult) + '\t')
            file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
            goodResult = 0
            badResult = 0
        elif (mode == "HEALTHY"):
            print("MTCNN: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + " MTCNN" + "\\"
            pathCore = pathCore.replace(":", " ")
            os.mkdir(pathCore)
            pathGood = pathCore + "Dobre\\"
            pathBad = pathCore + "Zle\\"
            pathGoodBad = pathCore + "Dobre_Nietrafione\\"
            pathBadBad = pathCore + "Zle_Nietrafione\\"

            os.mkdir(pathGood)
            os.mkdir(pathBad)
            os.mkdir(pathGoodBad)
            os.mkdir(pathBadBad)
            ######################################################################################
            cpathGood = pathCore + "Chore_Dobre\\"
            cpathBad = pathCore + "Chore_Zle\\"
            cpathGoodBad = pathCore + "Chore_Dobre_Nietrafione\\"
            cpathBadBad = pathCore + "Chore_Zle_Nietrafione\\"
            os.mkdir(cpathGood)
            os.mkdir(cpathBad)
            os.mkdir(cpathGoodBad)
            os.mkdir(cpathBadBad)
            lister_good = glob.glob("ProbkiBadawcze/OsobaChora/Dobre/*")
            # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
            lister_bad = glob.glob("ProbkiBadawcze/OsobaChora/Zle/*")

            # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
            counter = 0

            file.writelines("\n\npitch\t" + "\troll\t" "\tyaw\t" + "filename\n")
            for image in lister_good:
                print(image)
                print("Iteracja: " + str(counter))
                counter += 1
                # dlibFaceDetector(image, cpathGood, cpathGoodBad)
                getFace(image, values[0], values[1], cpathGood, cpathGoodBad)

                if printDetails:
                    printDetails = False
            printDetails = True

            truePositive += goodResult
            falseNegative += badResult
            goodResult = 0
            badResult = 0
            counter = 0
            for image in lister_bad:
                print(image)
                print("Iteracja: " + str(counter))
                counter += 1
                # dlibFaceDetector(image, cpathBadBad, cpathBad)
                getFace(image, values[0], values[1], cpathBadBad, cpathBad)

                if printDetails:
                    printDetails = False
            printDetails = True
            falsePositive += badResult
            trueNegative += goodResult
            goodResult = 0
            badResult = 0
            ######################################################################################
            file.writelines("\n\npitch\t" + "\troll\t" "\tyaw\t" + "filename\n")
            for i in range(11, 12, 1):

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    # dlibFaceDetector(image, pathGood, pathGoodBad)
                    getFace(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                truePositive += goodResult
                falseNegative += badResult
                goodResult = 0
                badResult = 0
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    # dlibFaceDetector(image, pathBadBad, pathBad)
                    # getFace(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                falsePositive += badResult
                trueNegative += goodResult
                goodResult = 0
                badResult = 0
        # file.writelines(
        #     getTime + "\tMTCNN: " + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
        #     str(falseNegative) + "\tfalsePositive:\t" + str(
        #         falsePositive) + "\ttrueNegative:\t" + str(
        #         trueNegative) + "\tTotal:\t" + str(
        #         truePositive + trueNegative + falsePositive + falseNegative))


def researchOrderer2(alghoritmName, mode, values, listergood):
    global printDetails
    global goodResult, badResult
    lister_good = glob.glob(listergood)

    global positiveLister
    global getXTime
    getTimeFolderPersons = datetime.datetime.now()
    getXTime = str(getTimeFolderPersons.strftime("%Y-%m-%d - %H-%M-%S"))
    if (alghoritmName == "HOG"):

        if (mode == "HEALTHY"):
            counter = 0
            for image in lister_good:
                print(image)
                print("Iteracja: " + str(counter))
                counter += 1
                dlibFaceDetector(image, 0, 0)
                if printDetails:
                    printDetails = False
            printDetails = True


def higestValue(entry):
    values = entry
    higest = values[0]
    for x in range(1, len(entry), 1):
        if (higest < values[x]):
            higest = values[x]
    return higest


def lowestValue(entry):
    values = entry
    lowest = values[0]
    for x in range(1, len(entry), 1):
        if (lowest > values[x]):
            lowest = values[x]
    return lowest


def toGray(colorimg):
    gray = cv2.cvtColor(colorimg, cv2.COLOR_BGR2GRAY)
    gray = rgb2gray(colorimg)
    return gray


def compare_images(imageA, imageB, title):
    s = ssim(imageA, imageB)
    return s


def resizeImagestToSameLevel(image1, image2):
    x1height, x1width = image1.shape[:2]
    x2height, x2width = image2.shape[:2]
    if ((x1height * x1width) > (x2height * x2width)):
        crop = cv2.resize(image1, (x2width, x2height), interpolation=cv2.INTER_AREA)

        # imutils.resize(image1, x2width, x2height)
        image1 = crop
    else:
        # cropp = imutils.resize(image2, x1width, x1height)
        crop = cv2.resize(image2, (x1width, x1height), interpolation=cv2.INTER_AREA)
        image2 = crop

    return image1, image2


# def gen_sift_features(gray_img):
#     sift = cv2[0]
#     features2d.SIFT_create()
#     # kp is the keypoints
#     #
#     # desc is the SIFT descriptors, they're 128-dimensional vectors
#     # that we can use for our final features
#     kp, desc = sift.detectAndCompute(gray_img, None)
#     return kp, desc


def show_sift_features(gray_img, color_img, kp):
    cv2.imshow("nopper", cv2.drawKeypoints(gray_img, kp, color_img.copy()))
    cv2.waitKey(0)


def preprecessCorrectedFaceParts(aData, isSickCollection, path):
    global getXTime
    getTimeFolderPersons = datetime.datetime.now()
    pathCore = path + getXTime + " EXTRACTED DATA " + "\\"
    getXTime = str(getTimeFolderPersons.strftime("%Y-%m-%d - %H-%M-%S"))
    pathCore = pathCore.replace(":", " ")
    pathGood = pathCore + "Dobre\\"
    pathBad = pathCore + "Zle\\"
    os.mkdir(pathCore)
    os.mkdir(pathGood)
    os.mkdir(pathBad)
    outputDataArray = []

    for member in aData:
        member.printer()

        cleanImage = member.alignedImage.copy()
        # member.alignedImage
        height, width = member.alignedImage.shape[:2]
        print(height, width)
        fp0 = [member.shape[0][0], member.shape[0][1]]
        fp1 = [member.shape[1][0], member.shape[1][1]]
        fp2 = [member.shape[2][0], member.shape[2][1]]
        fp3 = [member.shape[3][0], member.shape[3][1]]
        fp4 = [member.shape[4][0], member.shape[4][1]]
        fp5 = [member.shape[5][0], member.shape[5][1]]
        fp6 = [member.shape[6][0], member.shape[6][1]]
        fp7 = [member.shape[7][0], member.shape[7][1]]
        fp8 = [member.shape[8][0], member.shape[8][1]]
        fp9 = [member.shape[9][0], member.shape[9][1]]
        fp10 = [member.shape[10][0], member.shape[10][1]]
        fp11 = [member.shape[11][0], member.shape[11][1]]
        fp12 = [member.shape[12][0], member.shape[12][1]]
        fp13 = [member.shape[13][0], member.shape[13][1]]
        fp14 = [member.shape[14][0], member.shape[14][1]]
        fp15 = [member.shape[15][0], member.shape[15][1]]
        fp16 = [member.shape[16][0], member.shape[16][1]]
        fp17 = [member.shape[17][0], member.shape[17][1]]  # poczatek lewej brwi
        fp18 = [member.shape[18][0], member.shape[18][1]]
        fp19 = [member.shape[19][0], member.shape[19][1]]
        fp20 = [member.shape[20][0], member.shape[20][1]]
        fp21 = [member.shape[21][0], member.shape[21][1]]
        fp22 = [member.shape[22][0], member.shape[22][1]]
        fp23 = [member.shape[23][0], member.shape[23][1]]
        fp24 = [member.shape[24][0], member.shape[24][1]]
        fp25 = [member.shape[25][0], member.shape[25][1]]
        fp26 = [member.shape[26][0], member.shape[26][1]]  # poczatek prawej brwi
        fp27 = [member.shape[27][0], member.shape[27][1]]  # pierwszy nosowy
        fp28 = [member.shape[28][0], member.shape[28][1]]  # drugi nosowy
        fp29 = [member.shape[29][0], member.shape[29][1]]  # ponad czubkiemn nosa
        fp30 = [member.shape[30][0], member.shape[30][1]]  # czubek nosa
        fp31 = [member.shape[31][0], member.shape[31][1]]
        fp32 = [member.shape[32][0], member.shape[32][1]]
        fp33 = [member.shape[33][0], member.shape[33][1]]
        fp34 = [member.shape[34][0], member.shape[34][1]]
        fp35 = [member.shape[35][0], member.shape[35][1]]
        fp36 = [member.shape[36][0], member.shape[36][1]]  # lewy zewnętrzny kącik oka
        fp37 = [member.shape[37][0], member.shape[37][1]]  # lewa gorna zewnetrzna zrenica
        fp38 = [member.shape[38][0], member.shape[38][1]]  # lewa gorna wewnetrzna zrenica
        fp39 = [member.shape[39][0], member.shape[39][1]]  # lewy wewnetrzny kącik oka
        fp40 = [member.shape[40][0], member.shape[40][1]]  # lewa dolna zrenica wewnetrzna
        fp41 = [member.shape[41][0], member.shape[41][1]]  # lewa dolna zrenica zewnetrzna
        fp42 = [member.shape[42][0], member.shape[42][1]]  # prawy wewnetrzny kacik oka
        fp43 = [member.shape[43][0], member.shape[43][1]]
        fp44 = [member.shape[44][0], member.shape[44][1]]
        fp45 = [member.shape[45][0], member.shape[45][1]]  # prawy zewnetrzny kacik oka
        fp46 = [member.shape[46][0], member.shape[46][1]]  # zewnetrzna prawwa zrenica
        fp47 = [member.shape[47][0], member.shape[47][1]]  # wewnetrzna prawa zrenica
        fp48 = [member.shape[48][0], member.shape[48][1]]  # lewy kącik ust
        fp49 = [member.shape[49][0], member.shape[49][1]]  # lewa warga zewnątrz góra
        fp50 = [member.shape[50][0], member.shape[50][1]]
        fp51 = [member.shape[51][0], member.shape[51][1]]  # środek górnej wargi
        fp52 = [member.shape[52][0], member.shape[52][1]]
        fp53 = [member.shape[53][0], member.shape[53][1]]  # prawa warga zewnątrz góra
        fp54 = [member.shape[54][0], member.shape[54][1]]  # prawy kącik ust
        fp55 = [member.shape[55][0], member.shape[55][1]]  # prawa warga zewnątrz dół
        fp56 = [member.shape[56][0], member.shape[56][1]]  # środek dolnej wargi
        fp57 = [member.shape[57][0], member.shape[57][1]]
        fp58 = [member.shape[58][0], member.shape[58][1]]
        fp59 = [member.shape[59][0], member.shape[59][1]]  # lewa warga dół zewnątrz
        fp60 = [member.shape[60][0], member.shape[60][1]]
        fp61 = [member.shape[61][0], member.shape[61][1]]
        fp62 = [member.shape[62][0], member.shape[62][1]]
        fp63 = [member.shape[63][0], member.shape[63][1]]
        fp64 = [member.shape[64][0], member.shape[64][1]]
        fp65 = [member.shape[65][0], member.shape[65][1]]
        fp66 = [member.shape[66][0], member.shape[66][1]]
        fp67 = [member.shape[67][0], member.shape[67][1]]

        # wyznaczanie kwadratów :

        # lewa fałdka nosa
        pointThreeLineY = 0
        pointThreeLineX = 0
        upperMounthMiddleLength = int(math.sqrt(math.pow(fp49[0] - fp51[0], 2) + (math.pow(fp49[1] - fp51[1], 2))))
        lowerMounthMiddleLength = int(math.sqrt(math.pow(fp59[0] - fp56[0], 2) + (math.pow(fp59[1] - fp56[1], 2))))
        if (upperMounthMiddleLength > lowerMounthMiddleLength):
            pointThreeLineY = fp49[0]
        else:
            pointThreeLineY = fp59[0]

        pointOne = [fp36[0], fp30[1]]

        distanceBetweenNewPointandEyeEdge = int(
            math.sqrt(math.pow(pointOne[0] - fp36[0], 2) + (math.pow(pointOne[1] - fp36[1], 2))))
        distanceBetweenEyeErisEdges = int(
            math.sqrt(math.pow(fp41[0] - fp40[0], 2) + (math.pow(fp41[1] - fp40[1], 2))))
        if (distanceBetweenNewPointandEyeEdge < distanceBetweenEyeErisEdges):
            pointOne[1] += int(math.sqrt(math.pow(fp41[0] - fp40[0], 2) + (math.pow(fp41[1] - fp40[1], 2))))

        pointW = pointThreeLineY - pointOne[0]

        pointThreeLineX = int(fp59[1] + math.sqrt(math.pow(fp59[0] - fp49[0], 2) + (math.pow(fp59[1] - fp49[1], 2))))
        pointH = pointThreeLineX - pointOne[1]
        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (255, 255, 0), 2)
        leftNosePart = cropped
        # cv2.imshow("nopper", member.alignedImage)
        # cv2.waitKey(0)

        #####################################################
        #####################################################
        # prawa fałdka nosa #####################################################
        #####################################################
        #####################################################
        poczatekKwadratuX = 0
        pointThreeLineY = 0
        upperMounthMiddleLength = int(math.sqrt(math.pow(fp53[0] - fp51[0], 2) + (math.pow(fp53[1] - fp51[1], 2))))
        lowerMounthMiddleLength = int(math.sqrt(math.pow(fp55[0] - fp56[0], 2) + (math.pow(fp55[1] - fp56[1], 2))))
        if (upperMounthMiddleLength > lowerMounthMiddleLength):
            poczatekKwadratuX = fp53[0]
        else:
            poczatekKwadratuX = fp55[0]

        pointTwo = [fp45[0], fp30[1]]

        distanceBetweenNewPointandEyeEdge = int(
            math.sqrt(math.pow(pointTwo[0] - fp45[0], 2) + (math.pow(pointTwo[1] - fp45[1], 2))))
        distanceBetweenEyeErisEdges = int(
            math.sqrt(math.pow(fp46[0] - fp47[0], 2) + (math.pow(fp46[1] - fp47[1], 2))))
        if (distanceBetweenNewPointandEyeEdge < distanceBetweenEyeErisEdges):
            pointTwo[1] += int(math.sqrt(math.pow(fp46[0] - fp47[0], 2) + (math.pow(fp46[1] - fp47[1], 2))))

        pointW = pointTwo[0] - poczatekKwadratuX
        pointOne = [poczatekKwadratuX, pointTwo[1]]

        pointThreeLineY = int(fp55[1] + math.sqrt(math.pow(fp55[0] - fp53[0], 2) + (math.pow(fp55[1] - fp53[1], 2))))
        pointH = pointThreeLineY - pointOne[1]
        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (255, 255, 0), 2)
        rightNosePart = cropped
        # cv2.imshow("nopper", member.alignedImage)
        # cv2.waitKey(0)

        ##################################### lewy kącik ust 49 / 59
        upperMounthMiddleLength = int(math.sqrt(math.pow(fp49[0] - fp51[0], 2) + (math.pow(fp49[1] - fp51[1], 2))))
        lowerMounthMiddleLength = int(math.sqrt(math.pow(fp59[0] - fp56[0], 2) + (math.pow(fp59[1] - fp56[1], 2))))
        if (upperMounthMiddleLength > lowerMounthMiddleLength):
            pointTwooLineX = fp49[0]

        else:
            pointTwooLineX = fp59[0]

        pointTwo = [pointTwooLineX,
                    fp49[1] - int(0.5 * (math.sqrt(math.pow(fp49[0] - fp59[0], 2) + (math.pow(fp49[1] - fp59[1], 2)))))]

        pointH = int(2 * (math.sqrt(math.pow(fp49[0] - fp59[0], 2) + (math.pow(fp49[1] - fp59[1], 2)))))
        # # wyznaczanie odleglosci w
        middlepoint = [int((fp49[0] + fp59[0]) / 2), int((fp49[1] + fp59[1]) / 2)]
        pointW = int(1.5 * (math.sqrt(math.pow(middlepoint[0] - fp48[0], 2) + (math.pow(middlepoint[1] - fp48[1], 2)))))
        pointOne = [pointTwo[0] - pointW, pointTwo[1]]
        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (255, 140, 0), 2)
        leftMounthEdge = cropped
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)

        # prawy kącik ust 53 / 59 ###############
        upperMounthMiddleLength = int(math.sqrt(math.pow(fp53[0] - fp51[0], 2) + (math.pow(fp53[1] - fp51[1], 2))))
        lowerMounthMiddleLength = int(math.sqrt(math.pow(fp55[0] - fp56[0], 2) + (math.pow(fp55[1] - fp56[1], 2))))
        if (upperMounthMiddleLength < lowerMounthMiddleLength):
            pointTwooLineX = fp53[0]

        else:
            pointTwooLineX = fp55[0]

        pointOne = [pointTwooLineX,
                    fp53[1] - int(0.5 * (math.sqrt(math.pow(fp53[0] - fp55[0], 2) + (math.pow(fp53[1] - fp55[1], 2)))))]

        pointH = int(2 * (math.sqrt(math.pow(fp53[0] - fp55[0], 2) + (math.pow(fp53[1] - fp55[1], 2)))))
        # # wyznaczanie odleglosci w
        middlepoint = [int((fp53[0] + fp55[0]) / 2), int((fp53[1] + fp55[1]) / 2)]
        pointW = int(1.5 * (math.sqrt(math.pow(middlepoint[0] - fp54[0], 2) + (math.pow(middlepoint[1] - fp54[1], 2)))))
        # pointOne = [pointTwo[0] - pointW, pointTwo[1]]
        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (255, 140, 0), 2)
        rightMounthEdge = cropped
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)

        # lewy kącik oka ##################################

        pointOne = fp17
        pointW = fp36[0] - fp17[0]
        pointH = int(1.5 * math.sqrt(math.pow(fp17[0] - fp36[0], 2) + (math.pow(fp17[1] - fp36[1], 2))))

        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]
        leftEyeEdge = cropped

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (50, 140, 170), 2)
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)
        # prawy kącik oka ##################################
        pointOne = [fp45[0], fp26[1]]
        pointW = fp26[0] - pointOne[0]
        pointH = int(1.5 * math.sqrt(math.pow(fp45[0] - fp26[0], 2) + (math.pow(fp45[1] - fp26[1], 2))))

        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]
        rightEyeEdge = cropped

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (50, 140, 255), 2)
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)

        # lewa podpowieka ##################################
        slope = math.atan2(fp30[1] - fp39[1], fp39[0] - fp30[0]) * (180.0 / math.pi)
        pointOne = [fp36[0], 0]
        pointW = int(math.sqrt(math.pow(fp39[0] - fp36[0], 2) + (math.pow(fp39[1] - fp36[1], 2))))
        if (slope > 90):
            slope = math.fabs(slope - 180)

        if (slope < 22):
            pointH = 2 * int(math.sqrt(math.pow(fp29[0] - fp30[0], 2) + (math.pow(fp29[1] - fp30[1], 2))))

        else:
            pointH = int(1.5 * (math.sqrt(math.pow(fp29[0] - fp30[0], 2) + (math.pow(fp29[1] - fp30[1], 2)))))

        pointOne[1] = higestValue([fp36[1], fp40[1], fp39[1], fp41[1]])

        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]
        leftUnderEye = cropped

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (50, 240, 70), 2)

        print("slope: " + str(slope))
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)

        # prawa podpowieka ##################################
        slope = math.atan2(fp30[1] - fp42[1], fp42[0] - fp30[0]) * (180.0 / math.pi)
        pointOne = [fp42[0], 0]
        pointW = int(math.sqrt(math.pow(fp42[0] - fp45[0], 2) + (math.pow(fp42[1] - fp45[1], 2))))

        if (slope > 90):
            slope = math.fabs(slope - 180)

        if (slope < 22):
            pointH = 2 * int(math.sqrt(math.pow(fp29[0] - fp30[0], 2) + (math.pow(fp29[1] - fp30[1], 2))))

        else:
            pointH = int(1.5 * (math.sqrt(math.pow(fp29[0] - fp30[0], 2) + (math.pow(fp29[1] - fp30[1], 2)))))

        pointOne[1] = higestValue([fp42[1], fp46[1], fp47[1], fp45[1]])

        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]
        rightUnderEye = cropped

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (50, 240, 70), 2)

        print("slope: " + str(slope))
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)

        # Usta ###############

        lowestMounthLevel = higestValue([fp48[1], fp54[1], fp55[1], fp56[1], fp57[1], fp58[1], fp59[1], fp60[1]])
        highestMounthLevel = lowestValue([fp48[1], fp49[1], fp50[1], fp51[1], fp52[1], fp53[1], fp54[1]])
        pointH = lowestMounthLevel - highestMounthLevel
        pointW = fp54[0] - fp48[0]
        pointOne = [fp48[0], highestMounthLevel]
        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]
        mounthMainArea = cropped
        mounthLeftPart = mounthMainArea[0:0 + pointH, 0:0 + int(pointW / 2)]
        mounthRightPart = mounthMainArea[0:0 + pointH, int(pointW / 2):int(pointW / 2) + pointW]
        # cv2.imshow("nope", mounthLeftPart)
        # cv2.waitKey(0)
        # cv2.imshow("nope", mounthRightPart)
        # cv2.waitKey(0)

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (255, 0, 60), 2)
        # cv2.imshow("nope", member.alignedImage)
        # cv2.waitKey(0)
        ##########################################
        ##########################################
        # rotowanie  :)
        ##########################################
        ##########################################
        # leftNosePart
        rightNosePartREV = cv2.flip(rightNosePart, 1)
        # leftMounthEdge
        rightMounthEdgeREV = cv2.flip(rightMounthEdge, 1)
        # leftEyeEdge
        rightEyeEdgeREV = cv2.flip(rightEyeEdge, 1)
        # leftUnderEye
        rightUnderEyeREV = cv2.flip(rightUnderEye, 1)
        # mounthLeftPart
        mounthRightPartREV = cv2.flip(mounthRightPart, 1)
        ##########################################
        ##########################################
        # dobieranie rozmiaru   :)
        ##########################################
        ##########################################
        leftNosePartGRAY = toGray(leftNosePart)
        rightNosePartREVGRAY = toGray(rightNosePartREV)
        leftMounthEdgeGRAY = toGray(leftMounthEdge)
        rightMounthEdgeREVGRAY = toGray(rightMounthEdgeREV)
        leftEyeEdgeGRAY = toGray(leftEyeEdge)
        rightEyeEdgeREVGRAY = toGray(rightEyeEdgeREV)
        leftUnderEyeGRAY = toGray(leftUnderEye)
        rightUnderEyeREVGRAY = toGray(rightUnderEyeREV)
        mounthLeftPartGRAY = toGray(mounthLeftPart)
        mounthRightPartREVGRAY = toGray(mounthRightPartREV)

        leftNosePartGRAY, rightNosePartREVGRAY = resizeImagestToSameLevel(leftNosePartGRAY, rightNosePartREVGRAY)
        leftMounthEdgeGRAY, rightMounthEdgeREVGRAY = resizeImagestToSameLevel(leftMounthEdgeGRAY,
                                                                              rightMounthEdgeREVGRAY)
        leftEyeEdgeGRAY, rightEyeEdgeREVGRAY = resizeImagestToSameLevel(leftEyeEdgeGRAY, rightEyeEdgeREVGRAY)
        leftUnderEyeGRAY, rightUnderEyeREVGRAY = resizeImagestToSameLevel(leftUnderEyeGRAY, rightUnderEyeREVGRAY)
        mounthLeftPartGRAY, mounthRightPartREVGRAY = resizeImagestToSameLevel(mounthLeftPartGRAY,
                                                                              mounthRightPartREVGRAY)

        analysedParts = [[leftNosePartGRAY, rightNosePartREVGRAY], [leftMounthEdgeGRAY, rightMounthEdgeREVGRAY],
                         [leftEyeEdgeGRAY, rightEyeEdgeREVGRAY], [leftUnderEyeGRAY, rightUnderEyeREVGRAY],
                         [mounthLeftPartGRAY, mounthRightPartREVGRAY]]

        singleReturnArray = []
        for singleComparasion in analysedParts:
            singleReturnArray.append(compare_images(singleComparasion[0], singleComparasion[1], "placki"))

        if (isSickCollection == 0 or isSickCollection == 3):
            singleReturnArray.append(0)
        else:
            singleReturnArray.append(1)
        outputDataArray.append(singleReturnArray)

        cv2.imwrite(pathCore + pathlib.Path(member.fileName).name, member.alignedImage)
    print("dlugosc wektora z wydobytymi danymi: " + str(len(outputDataArray)))
    for x in outputDataArray:
        print(x)

    if isSickCollection == 1:
        # if not (pathlib.Path("ExposedData_Sick.txt").is_file()):
        with open('ExposedData_Sick.txt', 'w') as filehandle:
            json.dump(outputDataArray, filehandle)

    elif (isSickCollection == 0):
        # if not (pathlib.Path("ExposedData_Healthy.txt").is_file()):
        with open('ExposedData_Healthy.txt', 'w') as filehandle:
            json.dump(outputDataArray, filehandle)
    elif (isSickCollection == 2):
        # if not (pathlib.Path("ExposedData_Healthy.txt").is_file()):
        with open('ExposedData_Healthy_test.txt', 'w') as filehandle:
            json.dump(outputDataArray, filehandle)
    elif (isSickCollection == 3):
        # if not (pathlib.Path("ExposedData_Healthy.txt").is_file()):
        with open('ExposedData_Sick_test.txt', 'w') as filehandle:
            json.dump(outputDataArray, filehandle)

    aData = []


def extractData():
    global positiveData, negativeData, positiveDataTest, negativeDataTest
    with open("ExposedData_Sick.txt", "r")as filehandle:
        positiveData = json.load(filehandle)

    print(positiveData)

    with open('ExposedData_Healthy.txt', 'r') as filehandle:
        negativeData = json.load(filehandle)
    print(negativeData)
    #######################
    with open('ExposedData_Sick_test.txt', 'r') as filehandle:
        positiveDataTest = json.load(filehandle)
    print(positiveDataTest)
    with open('ExposedData_Healthy_test.txt', 'r') as filehandle:
        negativeDataTest = json.load(filehandle)
    print(negativeDataTest)


def kNearestClassifierLarning():
    totalData = positiveData + negativeData
    totalLength = len(totalData)
    totalData = np.array(totalData)
    # iris = datasets.load_iris()
    # features_train, features_test, labels_train, labels_test = train_test_split(iris.data, iris.target, test_size=0.25)
    X = totalData[:, range(0, 5)]
    Y = totalData[:, 5]
    knn = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto')
    knn.fit(X, Y)

    totalTestData = negativeDataTest + positiveDataTest
    totalTestData = np.array(totalTestData)
    testData = totalTestData[:, range(0, 5)]
    testDataLabels = totalTestData[:, 5]
    knnPredictor = knn.predict(testData)

    output = accuracy_score(testDataLabels, knnPredictor)
    print("output accuracy: " + str(output))


def findAngle(p0, p1, p2):
    a = math.pow(p1[0] - p0[0], 2) + math.pow(p1[1] - p0[1], 2)
    b = math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2)
    c = math.pow(p2[0] - p0[0], 2) + math.pow(p2[1] - p0[1], 2)
    return math.acos((a + b - c) / math.sqrt(4 * a * b)) * (180 / math.pi)
    # return math.acos((a + b - c) / math.sqrt(4 * a * b))  # * (180 / math.pi)


def gaussianNaiveBayessLearning():
    gnb = GaussianNB()
    totalData = positiveData + negativeData
    totalLength = len(totalData)
    totalData = np.array(totalData)
    X = totalData[:, range(0, 5)]
    Y = totalData[:, 5]

    totalTestData = negativeDataTest + positiveDataTest
    totalTestData = np.array(totalTestData)
    testData = totalTestData[:, range(0, 5)]
    testDataLabels = totalTestData[:, 5]
    y_pred = gnb.fit(X, Y).predict(testData)
    print("Number of mislabeled points out of a total %d points : %d" % (
        len(totalTestData), (testData != y_pred).sum()))


def intersectionPointFinder(pointL1A, pointL1B, pointL2A, pointL2B):
    # xOutput = ((((x1 * y2) - (y1 * x2)) * (x3 - x4)) - ((x1 - x2) * ((x3 * y4) - y3 * x4))) / (
    #         ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))
    #
    # yOutput = ((((x1 * y2) - (y1 * x2)) * (y3 - y4)) - ((y1 - y2) * ((x3 * y4) - y3 * x4))) / (
    #         ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4)))

    intersectionCheck = ((pointL1A[0] - pointL1B[0]) * (pointL2A[1] - pointL2B[1])) - (
            (pointL1A[1] - pointL1B[1]) * (pointL2A[0] - pointL2B[0]))
    if intersectionCheck == 0:
        return 0, 0

    xOutput = ((((pointL1A[0] * pointL1B[1]) - (pointL1A[1] * pointL1B[0])) * (pointL2A[0] - pointL2B[0])) - (
            (pointL1A[0] - pointL1B[0]) * ((pointL2A[0] * pointL2B[1]) - pointL2A[1] * pointL2B[0]))) / (
                      ((pointL1A[0] - pointL1B[0]) * (pointL2A[1] - pointL2B[1])) - (
                      (pointL1A[1] - pointL1B[1]) * (pointL2A[0] - pointL2B[0])))

    yOutput = ((((pointL1A[0] * pointL1B[1]) - (pointL1A[1] * pointL1B[0])) * (pointL2A[1] - pointL2B[1])) - (
            (pointL1A[1] - pointL1B[1]) * ((pointL2A[0] * pointL2B[1]) - pointL2A[1] * pointL2B[0]))) / (
                      ((pointL1A[0] - pointL1B[0]) * (pointL2A[1] - pointL2B[1])) - (
                      (pointL1A[1] - pointL1B[1]) * (pointL2A[0] - pointL2B[0])))

    print("Intersection Point:\nX= %d Y= %d" % (xOutput, yOutput))
    return xOutput, yOutput


def preprecessCorrectedFaceAngles(aData, isSickCollection, path):
    # wyliczenie funkcji 4 prostych na podstawie ich punktów
    outputDataArray = []
    for member in aData:
        member.printer()
        height, width = member.alignedImage.shape[:2]
        print(height, width)
        fp0 = [member.shape[0][0], member.shape[0][1]]
        fp1 = [member.shape[1][0], member.shape[1][1]]
        fp2 = [member.shape[2][0], member.shape[2][1]]
        fp3 = [member.shape[3][0], member.shape[3][1]]
        fp4 = [member.shape[4][0], member.shape[4][1]]
        fp5 = [member.shape[5][0], member.shape[5][1]]
        fp6 = [member.shape[6][0], member.shape[6][1]]
        fp7 = [member.shape[7][0], member.shape[7][1]]
        fp8 = [member.shape[8][0], member.shape[8][1]]
        fp9 = [member.shape[9][0], member.shape[9][1]]
        fp10 = [member.shape[10][0], member.shape[10][1]]
        fp11 = [member.shape[11][0], member.shape[11][1]]
        fp12 = [member.shape[12][0], member.shape[12][1]]
        fp13 = [member.shape[13][0], member.shape[13][1]]
        fp14 = [member.shape[14][0], member.shape[14][1]]
        fp15 = [member.shape[15][0], member.shape[15][1]]
        fp16 = [member.shape[16][0], member.shape[16][1]]
        fp17 = [member.shape[17][0], member.shape[17][1]]  # poczatek lewej brwi
        fp18 = [member.shape[18][0], member.shape[18][1]]
        fp19 = [member.shape[19][0], member.shape[19][1]]
        fp20 = [member.shape[20][0], member.shape[20][1]]
        fp21 = [member.shape[21][0], member.shape[21][1]]
        fp22 = [member.shape[22][0], member.shape[22][1]]
        fp23 = [member.shape[23][0], member.shape[23][1]]
        fp24 = [member.shape[24][0], member.shape[24][1]]
        fp25 = [member.shape[25][0], member.shape[25][1]]
        fp26 = [member.shape[26][0], member.shape[26][1]]  # poczatek prawej brwi
        fp27 = [member.shape[27][0], member.shape[27][1]]  # pierwszy nosowy
        fp28 = [member.shape[28][0], member.shape[28][1]]  # drugi nosowy
        fp29 = [member.shape[29][0], member.shape[29][1]]  # ponad czubkiemn nosa
        fp30 = [member.shape[30][0], member.shape[30][1]]  # czubek nosa
        fp31 = [member.shape[31][0], member.shape[31][1]]
        fp32 = [member.shape[32][0], member.shape[32][1]]
        fp33 = [member.shape[33][0], member.shape[33][1]]
        fp34 = [member.shape[34][0], member.shape[34][1]]
        fp35 = [member.shape[35][0], member.shape[35][1]]
        fp36 = [member.shape[36][0], member.shape[36][1]]  # lewy zewnętrzny kącik oka
        fp37 = [member.shape[37][0], member.shape[37][1]]  # lewa gorna zewnetrzna zrenica
        fp38 = [member.shape[38][0], member.shape[38][1]]  # lewa gorna wewnetrzna zrenica
        fp39 = [member.shape[39][0], member.shape[39][1]]  # lewy wewnetrzny kącik oka
        fp40 = [member.shape[40][0], member.shape[40][1]]  # lewa dolna zrenica wewnetrzna
        fp41 = [member.shape[41][0], member.shape[41][1]]  # lewa dolna zrenica zewnetrzna
        fp42 = [member.shape[42][0], member.shape[42][1]]  # prawy wewnetrzny kacik oka
        fp43 = [member.shape[43][0], member.shape[43][1]]
        fp44 = [member.shape[44][0], member.shape[44][1]]
        fp45 = [member.shape[45][0], member.shape[45][1]]  # prawy zewnetrzny kacik oka
        fp46 = [member.shape[46][0], member.shape[46][1]]  # zewnetrzna prawwa zrenica
        fp47 = [member.shape[47][0], member.shape[47][1]]  # wewnetrzna prawa zrenica
        fp48 = [member.shape[48][0], member.shape[48][1]]  # lewy kącik ust
        fp49 = [member.shape[49][0], member.shape[49][1]]  # lewa warga zewnątrz góra
        fp50 = [member.shape[50][0], member.shape[50][1]]
        fp51 = [member.shape[51][0], member.shape[51][1]]  # środek górnej wargi
        fp52 = [member.shape[52][0], member.shape[52][1]]
        fp53 = [member.shape[53][0], member.shape[53][1]]  # prawa warga zewnątrz góra
        fp54 = [member.shape[54][0], member.shape[54][1]]  # prawy kącik ust
        fp55 = [member.shape[55][0], member.shape[55][1]]  # prawa warga zewnątrz dół
        fp56 = [member.shape[56][0], member.shape[56][1]]  # środek dolnej wargi
        fp57 = [member.shape[57][0], member.shape[57][1]]
        fp58 = [member.shape[58][0], member.shape[58][1]]
        fp59 = [member.shape[59][0], member.shape[59][1]]  # lewa warga dół zewnątrz
        fp60 = [member.shape[60][0], member.shape[60][1]]
        fp61 = [member.shape[61][0], member.shape[61][1]]
        fp62 = [member.shape[62][0], member.shape[62][1]]
        fp63 = [member.shape[63][0], member.shape[63][1]]
        fp64 = [member.shape[64][0], member.shape[64][1]]
        fp65 = [member.shape[65][0], member.shape[65][1]]
        fp66 = [member.shape[66][0], member.shape[66][1]]
        fp67 = [member.shape[67][0], member.shape[67][1]]

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # wyznaczanie miejsc przecięcia kazdej z lini@@@@@@@@@@@@@@@@
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # nosowa i oczna

        xNoseEyes, yNoseEyes = intersectionPointFinder(fp39, fp42, fp27, fp30)
        xNoseUnderNose, yNoseUnderNose = intersectionPointFinder(fp35, fp31, fp27, fp30)
        xNoseMounth, yNoseMounth = intersectionPointFinder(fp48, fp54, fp27, fp30)

        # print(math.atan2(fp27[1] - yNoseEyes, xNoseEyes - fp27[0]) * (180.0 / math.pi))
        # print(math.atan2(fp27[1] - yNoseUnderNose, xNoseUnderNose - fp27[0]) * (180.0 / math.pi))
        # print(math.atan2(fp27[1] - yNoseMounth, xNoseMounth - fp27[0]) * (180.0 / math.pi))
        print("@@@@@@@@@@@@@@@@@@@@@@@@")
        angle1 = float(findAngle(fp39, [xNoseEyes, yNoseEyes], fp27) / 180)
        angle2 = float(findAngle(fp31, [xNoseUnderNose, yNoseUnderNose], fp27) / 180)
        angle3 = float(findAngle(fp48, [xNoseMounth, yNoseMounth], fp27) / 180)
        print("@@@@@@@@@@@@@@@@@@@@@@@@")

        print("@@@@@@@@@@@@@@@@@@@@@@@@")
        angle4 = float(findAngle(fp49, fp48, fp59) / 180)  # lewy kącik
        angle5 = float(findAngle(fp53, fp54, fp55) / 180)  # prawy kacik
        angle6 = float(findAngle(fp27, fp48, fp56) / 180)  # lewy kacik i nos
        angle7 = float(findAngle(fp27, fp54, fp56) / 180)  # prawy kacik i nos

        # angle4 = float(findAngle(fp27, [xNoseEyes, yNoseEyes], fp42)/180)
        # angle5 = float(findAngle(fp27, [xNoseUnderNose, yNoseUnderNose], fp35)/180)
        # angle6 = float(findAngle(fp27, [xNoseMounth, yNoseMounth], fp54)/180)

        angler = [angle1, angle2, angle3, angle4, angle5, angle6, angle7]
        print(angler)
        # scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
        # scaler = scaler.fit(angler)
        # new = scaler.transform(data)
        # for x in new:
        #     print (str(x))

        if (isSickCollection == 0 or isSickCollection == 3):
            angler.append(0)
        else:
            angler.append(1)

        outputDataArray.append(angler)

    if isSickCollection == 1:
        # if not (pathlib.Path("ExposedData_Sick.txt").is_file()):
        with open('ExposedData_Angle_Sick.txt', 'w') as filehandle:
            json.dump(angler, filehandle)

    elif (isSickCollection == 0):
        # if not (pathlib.Path("ExposedData_Healthy.txt").is_file()):
        with open('ExposedData_Angle_Healthy.txt', 'w') as filehandle:
            json.dump(angler, filehandle)
    elif (isSickCollection == 2):
        # if not (pathlib.Path("ExposedData_Healthy.txt").is_file()):
        with open('ExposedData_Angle_Healthy_test.txt', 'w') as filehandle:
            json.dump(angler, filehandle)
    elif (isSickCollection == 3):
        # if not (pathlib.Path("ExposedData_Healthy.txt").is_file()):
        with open('ExposedData_Angle_Sick_test.txt', 'w') as filehandle:
            json.dump(angler, filehandle)

    aData = []


def preprecessCorrectedFacePartsAndAngles(aData, isSickCollection, path):
    global getXTime
    global analysedData
    getTimeFolderPersons = datetime.datetime.now()
    pathCore = path + getXTime + " EXTRACTED DATA " + "\\"
    getXTime = str(getTimeFolderPersons.strftime("%Y-%m-%d - %H-%M-%S"))
    pathCore = pathCore.replace(":", " ")
    pathGood = pathCore + "Dobre\\"
    pathBad = pathCore + "Zle\\"
    os.mkdir(pathCore)
    os.mkdir(pathGood)
    os.mkdir(pathBad)
    outputDataArray = []

    for member in aData:
        member.printer()

        cleanImage = member.alignedImage.copy()
        cleanImage = rgb2gray(cleanImage)
        # member.alignedImage
        height, width = member.alignedImage.shape[:2]
        print(height, width)
        fp0 = [member.shape[0][0], member.shape[0][1]]
        fp1 = [member.shape[1][0], member.shape[1][1]]
        fp2 = [member.shape[2][0], member.shape[2][1]]
        fp3 = [member.shape[3][0], member.shape[3][1]]
        fp4 = [member.shape[4][0], member.shape[4][1]]
        fp5 = [member.shape[5][0], member.shape[5][1]]
        fp6 = [member.shape[6][0], member.shape[6][1]]
        fp7 = [member.shape[7][0], member.shape[7][1]]
        fp8 = [member.shape[8][0], member.shape[8][1]]
        fp9 = [member.shape[9][0], member.shape[9][1]]
        fp10 = [member.shape[10][0], member.shape[10][1]]
        fp11 = [member.shape[11][0], member.shape[11][1]]
        fp12 = [member.shape[12][0], member.shape[12][1]]
        fp13 = [member.shape[13][0], member.shape[13][1]]
        fp14 = [member.shape[14][0], member.shape[14][1]]
        fp15 = [member.shape[15][0], member.shape[15][1]]
        fp16 = [member.shape[16][0], member.shape[16][1]]
        fp17 = [member.shape[17][0], member.shape[17][1]]  # poczatek lewej brwi
        fp18 = [member.shape[18][0], member.shape[18][1]]
        fp19 = [member.shape[19][0], member.shape[19][1]]
        fp20 = [member.shape[20][0], member.shape[20][1]]
        fp21 = [member.shape[21][0], member.shape[21][1]]
        fp22 = [member.shape[22][0], member.shape[22][1]]
        fp23 = [member.shape[23][0], member.shape[23][1]]
        fp24 = [member.shape[24][0], member.shape[24][1]]
        fp25 = [member.shape[25][0], member.shape[25][1]]
        fp26 = [member.shape[26][0], member.shape[26][1]]  # poczatek prawej brwi
        fp27 = [member.shape[27][0], member.shape[27][1]]  # pierwszy nosowy
        fp28 = [member.shape[28][0], member.shape[28][1]]  # drugi nosowy
        fp29 = [member.shape[29][0], member.shape[29][1]]  # ponad czubkiemn nosa
        fp30 = [member.shape[30][0], member.shape[30][1]]  # czubek nosa
        fp31 = [member.shape[31][0], member.shape[31][1]]
        fp32 = [member.shape[32][0], member.shape[32][1]]
        fp33 = [member.shape[33][0], member.shape[33][1]]
        fp34 = [member.shape[34][0], member.shape[34][1]]
        fp35 = [member.shape[35][0], member.shape[35][1]]
        fp36 = [member.shape[36][0], member.shape[36][1]]  # lewy zewnętrzny kącik oka
        fp37 = [member.shape[37][0], member.shape[37][1]]  # lewa gorna zewnetrzna zrenica
        fp38 = [member.shape[38][0], member.shape[38][1]]  # lewa gorna wewnetrzna zrenica
        fp39 = [member.shape[39][0], member.shape[39][1]]  # lewy wewnetrzny kącik oka
        fp40 = [member.shape[40][0], member.shape[40][1]]  # lewa dolna zrenica wewnetrzna
        fp41 = [member.shape[41][0], member.shape[41][1]]  # lewa dolna zrenica zewnetrzna
        fp42 = [member.shape[42][0], member.shape[42][1]]  # prawy wewnetrzny kacik oka
        fp43 = [member.shape[43][0], member.shape[43][1]]
        fp44 = [member.shape[44][0], member.shape[44][1]]
        fp45 = [member.shape[45][0], member.shape[45][1]]  # prawy zewnetrzny kacik oka
        fp46 = [member.shape[46][0], member.shape[46][1]]  # zewnetrzna prawwa zrenica
        fp47 = [member.shape[47][0], member.shape[47][1]]  # wewnetrzna prawa zrenica
        fp48 = [member.shape[48][0], member.shape[48][1]]  # lewy kącik ust
        fp49 = [member.shape[49][0], member.shape[49][1]]  # lewa warga zewnątrz góra
        fp50 = [member.shape[50][0], member.shape[50][1]]
        fp51 = [member.shape[51][0], member.shape[51][1]]  # środek górnej wargi
        fp52 = [member.shape[52][0], member.shape[52][1]]
        fp53 = [member.shape[53][0], member.shape[53][1]]  # prawa warga zewnątrz góra
        fp54 = [member.shape[54][0], member.shape[54][1]]  # prawy kącik ust
        fp55 = [member.shape[55][0], member.shape[55][1]]  # prawa warga zewnątrz dół
        fp56 = [member.shape[56][0], member.shape[56][1]]  # środek dolnej wargi
        fp57 = [member.shape[57][0], member.shape[57][1]]
        fp58 = [member.shape[58][0], member.shape[58][1]]
        fp59 = [member.shape[59][0], member.shape[59][1]]  # lewa warga dół zewnątrz
        fp60 = [member.shape[60][0], member.shape[60][1]]
        fp61 = [member.shape[61][0], member.shape[61][1]]
        fp62 = [member.shape[62][0], member.shape[62][1]]
        fp63 = [member.shape[63][0], member.shape[63][1]]
        fp64 = [member.shape[64][0], member.shape[64][1]]
        fp65 = [member.shape[65][0], member.shape[65][1]]
        fp66 = [member.shape[66][0], member.shape[66][1]]
        fp67 = [member.shape[67][0], member.shape[67][1]]

        # wyznaczanie kwadratów :

        # lewa fałdka nosa
        pointThreeLineY = 0
        pointThreeLineX = 0
        upperMounthMiddleLength = int(math.sqrt(math.pow(fp49[0] - fp51[0], 2) + (math.pow(fp49[1] - fp51[1], 2))))
        lowerMounthMiddleLength = int(math.sqrt(math.pow(fp59[0] - fp56[0], 2) + (math.pow(fp59[1] - fp56[1], 2))))
        if (upperMounthMiddleLength > lowerMounthMiddleLength):
            pointThreeLineY = fp49[0]
        else:
            pointThreeLineY = fp59[0]

        pointOne = [fp36[0], fp30[1]]

        distanceBetweenNewPointandEyeEdge = int(
            math.sqrt(math.pow(pointOne[0] - fp36[0], 2) + (math.pow(pointOne[1] - fp36[1], 2))))
        distanceBetweenEyeErisEdges = int(
            math.sqrt(math.pow(fp41[0] - fp40[0], 2) + (math.pow(fp41[1] - fp40[1], 2))))
        if (distanceBetweenNewPointandEyeEdge < distanceBetweenEyeErisEdges):
            pointOne[1] += int(math.sqrt(math.pow(fp41[0] - fp40[0], 2) + (math.pow(fp41[1] - fp40[1], 2))))

        pointW = pointThreeLineY - pointOne[0]

        pointThreeLineX = int(fp59[1] + math.sqrt(math.pow(fp59[0] - fp49[0], 2) + (math.pow(fp59[1] - fp49[1], 2))))
        pointH = pointThreeLineX - pointOne[1]
        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (255, 255, 0), 2)
        leftNosePart = cropped
        # cv2.imshow("nopper", member.alignedImage)
        # cv2.waitKey(0)

        #####################################################
        #####################################################
        # prawa fałdka nosa #####################################################
        #####################################################
        #####################################################
        poczatekKwadratuX = 0
        pointThreeLineY = 0
        upperMounthMiddleLength = int(math.sqrt(math.pow(fp53[0] - fp51[0], 2) + (math.pow(fp53[1] - fp51[1], 2))))
        lowerMounthMiddleLength = int(math.sqrt(math.pow(fp55[0] - fp56[0], 2) + (math.pow(fp55[1] - fp56[1], 2))))
        if (upperMounthMiddleLength > lowerMounthMiddleLength):
            poczatekKwadratuX = fp53[0]
        else:
            poczatekKwadratuX = fp55[0]

        pointTwo = [fp45[0], fp30[1]]

        distanceBetweenNewPointandEyeEdge = int(
            math.sqrt(math.pow(pointTwo[0] - fp45[0], 2) + (math.pow(pointTwo[1] - fp45[1], 2))))
        distanceBetweenEyeErisEdges = int(
            math.sqrt(math.pow(fp46[0] - fp47[0], 2) + (math.pow(fp46[1] - fp47[1], 2))))
        if (distanceBetweenNewPointandEyeEdge < distanceBetweenEyeErisEdges):
            pointTwo[1] += int(math.sqrt(math.pow(fp46[0] - fp47[0], 2) + (math.pow(fp46[1] - fp47[1], 2))))

        pointW = pointTwo[0] - poczatekKwadratuX
        pointOne = [poczatekKwadratuX, pointTwo[1]]

        pointThreeLineY = int(fp55[1] + math.sqrt(math.pow(fp55[0] - fp53[0], 2) + (math.pow(fp55[1] - fp53[1], 2))))
        pointH = pointThreeLineY - pointOne[1]
        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (255, 255, 0), 2)
        rightNosePart = cropped
        # cv2.imshow("nopper", member.alignedImage)
        # cv2.waitKey(0)

        ##################################### lewy kącik ust 49 / 59
        upperMounthMiddleLength = int(math.sqrt(math.pow(fp49[0] - fp51[0], 2) + (math.pow(fp49[1] - fp51[1], 2))))
        lowerMounthMiddleLength = int(math.sqrt(math.pow(fp59[0] - fp56[0], 2) + (math.pow(fp59[1] - fp56[1], 2))))
        if (upperMounthMiddleLength > lowerMounthMiddleLength):
            pointTwooLineX = fp49[0]

        else:
            pointTwooLineX = fp59[0]

        pointTwo = [pointTwooLineX,
                    fp49[1] - int(0.5 * (math.sqrt(math.pow(fp49[0] - fp59[0], 2) + (math.pow(fp49[1] - fp59[1], 2)))))]

        pointH = int(2 * (math.sqrt(math.pow(fp49[0] - fp59[0], 2) + (math.pow(fp49[1] - fp59[1], 2)))))
        # # wyznaczanie odleglosci w
        middlepoint = [int((fp49[0] + fp59[0]) / 2), int((fp49[1] + fp59[1]) / 2)]
        pointW = int(1.5 * (math.sqrt(math.pow(middlepoint[0] - fp48[0], 2) + (math.pow(middlepoint[1] - fp48[1], 2)))))
        pointOne = [pointTwo[0] - pointW, pointTwo[1]]
        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (255, 140, 0), 2)
        leftMounthEdge = cropped
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)

        # prawy kącik ust 53 / 59 ###############
        upperMounthMiddleLength = int(math.sqrt(math.pow(fp53[0] - fp51[0], 2) + (math.pow(fp53[1] - fp51[1], 2))))
        lowerMounthMiddleLength = int(math.sqrt(math.pow(fp55[0] - fp56[0], 2) + (math.pow(fp55[1] - fp56[1], 2))))
        if (upperMounthMiddleLength < lowerMounthMiddleLength):
            pointTwooLineX = fp53[0]

        else:
            pointTwooLineX = fp55[0]

        pointOne = [pointTwooLineX,
                    fp53[1] - int(0.5 * (math.sqrt(math.pow(fp53[0] - fp55[0], 2) + (math.pow(fp53[1] - fp55[1], 2)))))]

        pointH = int(2 * (math.sqrt(math.pow(fp53[0] - fp55[0], 2) + (math.pow(fp53[1] - fp55[1], 2)))))
        # # wyznaczanie odleglosci w
        middlepoint = [int((fp53[0] + fp55[0]) / 2), int((fp53[1] + fp55[1]) / 2)]
        pointW = int(1.5 * (math.sqrt(math.pow(middlepoint[0] - fp54[0], 2) + (math.pow(middlepoint[1] - fp54[1], 2)))))
        # pointOne = [pointTwo[0] - pointW, pointTwo[1]]
        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (255, 140, 0), 2)
        rightMounthEdge = cropped
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)

        # lewy kącik oka ##################################

        pointOne = fp17
        pointW = fp36[0] - fp17[0]
        pointH = int(1.5 * math.sqrt(math.pow(fp17[0] - fp36[0], 2) + (math.pow(fp17[1] - fp36[1], 2))))

        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]
        leftEyeEdge = cropped

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (50, 140, 170), 2)
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)
        # prawy kącik oka ##################################
        pointOne = [fp45[0], fp26[1]]
        pointW = fp26[0] - pointOne[0]
        pointH = int(1.5 * math.sqrt(math.pow(fp45[0] - fp26[0], 2) + (math.pow(fp45[1] - fp26[1], 2))))

        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]
        rightEyeEdge = cropped

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (50, 140, 255), 2)
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)

        # lewa podpowieka ##################################
        slope = math.atan2(fp30[1] - fp39[1], fp39[0] - fp30[0]) * (180.0 / math.pi)
        pointOne = [fp36[0], 0]
        pointW = int(math.sqrt(math.pow(fp39[0] - fp36[0], 2) + (math.pow(fp39[1] - fp36[1], 2))))
        if (slope > 90):
            slope = math.fabs(slope - 180)

        if (slope < 22):
            pointH = 2 * int(math.sqrt(math.pow(fp29[0] - fp30[0], 2) + (math.pow(fp29[1] - fp30[1], 2))))

        else:
            pointH = int(1.5 * (math.sqrt(math.pow(fp29[0] - fp30[0], 2) + (math.pow(fp29[1] - fp30[1], 2)))))

        pointOne[1] = higestValue([fp36[1], fp40[1], fp39[1], fp41[1]])

        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]
        leftUnderEye = cropped

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (50, 240, 70), 2)

        print("slope: " + str(slope))
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)

        # prawa podpowieka ##################################
        slope = math.atan2(fp30[1] - fp42[1], fp42[0] - fp30[0]) * (180.0 / math.pi)
        pointOne = [fp42[0], 0]
        pointW = int(math.sqrt(math.pow(fp42[0] - fp45[0], 2) + (math.pow(fp42[1] - fp45[1], 2))))

        if (slope > 90):
            slope = math.fabs(slope - 180)

        if (slope < 22):
            pointH = 2 * int(math.sqrt(math.pow(fp29[0] - fp30[0], 2) + (math.pow(fp29[1] - fp30[1], 2))))

        else:
            pointH = int(1.5 * (math.sqrt(math.pow(fp29[0] - fp30[0], 2) + (math.pow(fp29[1] - fp30[1], 2)))))

        pointOne[1] = higestValue([fp42[1], fp46[1], fp47[1], fp45[1]])

        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]
        rightUnderEye = cropped

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (50, 240, 70), 2)

        print("slope: " + str(slope))
        # cv2.imshow("nope", member.alignedImage)
        #
        # cv2.waitKey(0)

        # Usta ###############

        lowestMounthLevel = higestValue([fp48[1], fp54[1], fp55[1], fp56[1], fp57[1], fp58[1], fp59[1], fp60[1]])
        highestMounthLevel = lowestValue([fp48[1], fp49[1], fp50[1], fp51[1], fp52[1], fp53[1], fp54[1]])
        pointH = lowestMounthLevel - highestMounthLevel
        pointW = fp54[0] - fp48[0]
        pointOne = [fp48[0], highestMounthLevel]
        cropped = cleanImage[pointOne[1]:pointOne[1] + pointH, pointOne[0]:pointOne[0] + pointW]
        mounthMainArea = cropped
        mounthLeftPart = mounthMainArea[0:0 + pointH, 0:0 + int(pointW / 2)]
        mounthRightPart = mounthMainArea[0:0 + pointH, int(pointW / 2):int(pointW / 2) + pointW]
        # cv2.imshow("nope", mounthLeftPart)
        # cv2.waitKey(0)
        # cv2.imshow("nope", mounthRightPart)
        # cv2.waitKey(0)

        cv2.rectangle(member.alignedImage, (pointOne[0], pointOne[1]), (pointOne[0] + pointW, pointOne[1] + pointH),
                      (255, 0, 60), 2)
        # cv2.imshow("nope", member.alignedImage)
        # cv2.waitKey(0)
        ##########################################
        ##########################################
        # rotowanie  :)
        ##########################################
        ##########################################
        # leftNosePart
        rightNosePartREV = cv2.flip(rightNosePart, 1)
        # leftMounthEdge
        rightMounthEdgeREV = cv2.flip(rightMounthEdge, 1)
        # leftEyeEdge
        rightEyeEdgeREV = cv2.flip(rightEyeEdge, 1)
        # leftUnderEye
        rightUnderEyeREV = cv2.flip(rightUnderEye, 1)
        # mounthLeftPart
        mounthRightPartREV = cv2.flip(mounthRightPart, 1)
        ##########################################
        ##########################################
        # dobieranie rozmiaru   :)
        ##########################################
        ##########################################
        # leftNosePartGRAY = toGray(leftNosePart)
        # rightNosePartREVGRAY = toGray(rightNosePartREV)
        # leftMounthEdgeGRAY = toGray(leftMounthEdge)
        # rightMounthEdgeREVGRAY = toGray(rightMounthEdgeREV)
        # leftEyeEdgeGRAY = toGray(leftEyeEdge)
        # rightEyeEdgeREVGRAY = toGray(rightEyeEdgeREV)
        # leftUnderEyeGRAY = toGray(leftUnderEye)
        # rightUnderEyeREVGRAY = toGray(rightUnderEyeREV)
        # mounthLeftPartGRAY = toGray(mounthLeftPart)
        # mounthRightPartREVGRAY = toGray(mounthRightPartREV)

        leftNosePartGRAY, rightNosePartREVGRAY = resizeImagestToSameLevel(leftNosePart, rightNosePartREV)
        leftMounthEdgeGRAY, rightMounthEdgeREVGRAY = resizeImagestToSameLevel(leftMounthEdge,
                                                                              rightMounthEdgeREV)
        leftEyeEdgeGRAY, rightEyeEdgeREVGRAY = resizeImagestToSameLevel(leftEyeEdge, rightEyeEdgeREV)
        leftUnderEyeGRAY, rightUnderEyeREVGRAY = resizeImagestToSameLevel(leftUnderEye, rightUnderEyeREV)
        mounthLeftPartGRAY, mounthRightPartREVGRAY = resizeImagestToSameLevel(mounthLeftPart,
                                                                              mounthRightPartREV)

        analysedParts = [[leftNosePartGRAY, rightNosePartREVGRAY], [leftMounthEdgeGRAY, rightMounthEdgeREVGRAY],
                         [leftEyeEdgeGRAY, rightEyeEdgeREVGRAY], [leftUnderEyeGRAY, rightUnderEyeREVGRAY],
                         [mounthLeftPartGRAY, mounthRightPartREVGRAY]]
        ######################################################################################################################

        # core = imutils.resize(cleanImage,width=500)
        # cv2.imshow("pre", core)
        # core = toGray(core)
        # cv2.imshow("grayscale", core)
        # # cv2.imshow("1 pre", singlePair[1])
        # # cv2.waitKey(0)
        # g1 = filters.gaussian(core, 0.5, multichannel=False, mode='reflect')
        #
        #
        #
        # # cv2.imshow("0 post", g1)
        # # cv2.imshow("1 post", g2)
        #
        # gamma_corrected = exposure.adjust_gamma(core, 2)
        # cv2.imshow("overall", gamma_corrected)
        #
        # # Logarithmic
        # logarithmic_corrected = exposure.adjust_log(core, 1)
        # cv2.imshow("log", logarithmic_corrected)
        #
        # # Global equalize
        # img_rescale = exposure.equalize_hist(core)
        # cv2.imshow("global equalize", img_rescale)
        #
        # # Equalization
        # selem = disk(30)
        # img_eq = rank.equalize(core, selem=selem)
        # cv2.imshow("local equalize", img_eq)
        #
        # cv2.waitKey(0)

        ######################################################################################################################
        singleReturnArray = []
        for singleComparasion in analysedParts:
            singleReturnArray.append(compare_images(singleComparasion[0], singleComparasion[1], "placki"))

        ######################################################################################################################
        xNoseEyes, yNoseEyes = intersectionPointFinder(fp39, fp42, fp27, fp30)
        xNoseUnderNose, yNoseUnderNose = intersectionPointFinder(fp35, fp31, fp27, fp30)
        xNoseMounth, yNoseMounth = intersectionPointFinder(fp48, fp54, fp27, fp30)

        print("@@@@@@@@@@@@@@@@@@@@@@@@")
        angle1 = float(findAngle(fp39, [xNoseEyes, yNoseEyes], fp27) / 180)
        angle2 = float(findAngle(fp31, [xNoseUnderNose, yNoseUnderNose], fp27) / 180)
        angle3 = float(findAngle(fp48, [xNoseMounth, yNoseMounth], fp27) / 180)
        print("@@@@@@@@@@@@@@@@@@@@@@@@")

        print("@@@@@@@@@@@@@@@@@@@@@@@@")
        angle4 = float(findAngle(fp49, fp48, fp59) / 180)  # lewy kącik
        angle5 = float(findAngle(fp53, fp54, fp55) / 180)  # prawy kacik
        angle6 = float(findAngle(fp27, fp48, fp56) / 180)  # lewy kacik i nos
        angle7 = float(findAngle(fp27, fp54, fp56) / 180)  # prawy kacik i nos

        angler = [angle1, angle2, angle3, angle4, angle5, angle6, angle7]
        print(angler)
        singleReturnArrayFinal = singleReturnArray + angler

        #####################################################################################################################

        pointsTable = [fp27, fp28, fp29, fp30, fp31, fp32, fp33, fp34, fp35, fp36, fp37, fp38, fp39, fp40, fp41, fp42,
                       fp43, fp44, fp45, fp46, fp47, fp48, fp49, fp50, fp51, fp52, fp53, fp54, fp55, fp56, fp57, fp58,
                       fp59, fp60, fp61, fp62, fp63, fp64, fp65, fp66]
        for singlePoint in pointsTable:
            r1 = singlePoint[0] / width
            r2 = singlePoint[1] / height
            singleReturnArrayFinal.append(r1)
            singleReturnArrayFinal.append(r2)
        #####################################################################################################################
        if (isSickCollection == 0 or isSickCollection == 2):
            singleReturnArrayFinal.append(0)
        else:
            singleReturnArrayFinal.append(1)

        outputDataArray.append(singleReturnArrayFinal)

        cv2.imwrite(pathCore + pathlib.Path(member.fileName).name, member.alignedImage)
    print("dlugosc wektora z wydobytymi danymi: " + str(len(outputDataArray)))
    for x in outputDataArray:
        print(x)

    if isSickCollection == 1:  # chorzy uczacy
        # if not (pathlib.Path("ExposedData_Sick.txt").is_file()):
        with open('ExposedData_Sick_TOTAL.txt', 'w') as filehandle:
            json.dump(outputDataArray, filehandle)
            aData = []
    elif (isSickCollection == 0):  # zdrowi
        # if not (pathlib.Path("ExposedData_Healthy.txt").is_file()):
        with open('ExposedData_Healthy_TOTAL.txt', 'w') as filehandle:
            json.dump(outputDataArray, filehandle)
            aData = []
    elif (isSickCollection == 2):  # zdrowi testowi
        # if not (pathlib.Path("ExposedData_Healthy.txt").is_file()):
        with open('ExposedData_Healthy_TOTAL_test.txt', 'w') as filehandle:
            json.dump(outputDataArray, filehandle)
            aData = []
    elif (isSickCollection == 3):  # chorzy testowi
        # if not (pathlib.Path("ExposedData_Healthy.txt").is_file()):
        with open('ExposedData_Sick_TOTAL_HARD_test.txt', 'w') as filehandle:
            json.dump(outputDataArray, filehandle)
            aData = []
    elif (isSickCollection == 4):  # chorzy testowi
        # if not (pathlib.Path("ExposedData_Healthy.txt").is_file()):
        with open('ExposedData_Sick_TOTAL_SOFT_test.txt', 'w') as filehandle:
            json.dump(outputDataArray, filehandle)
            aData = []
    analysedData = []


#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# researchOrderer("HOG", "HEALTHY", 0, "Proby Etapu trzeciego/Uczacy/Negatywne/*")
# preprecessCorrectedFaceParts(analysedData, 0,"Proby Etapu trzeciego/Uczacy/Wyjsciowe/")
# #
# researchOrderer("HOG", "HEALTHY", 0, "Proby Etapu trzeciego/Uczacy/Pozytywne/*")
# preprecessCorrectedFaceParts(analysedData, 1,"Proby Etapu trzeciego/Uczacy/Wyjsciowe/")


# researchOrderer("HOG", "HEALTHY", 0, "Proby Etapu trzeciego/Testowy/Negatywne/*")
# preprecessCorrectedFaceParts(analysedData, 2, "Proby Etapu trzeciego/Uczacy/Wyjsciowe/")
# # #
# researchOrderer("HOG", "HEALTHY", 0, "Proby Etapu trzeciego/Testowy/Pozytywne/Ciezkie/*")
# preprecessCorrectedFaceParts(analysedData, 3, "Proby Etapu trzeciego/Uczacy/Wyjsciowe/")

# extractData()
#
# kNearestClassifierLarning()

# gaussianNaiveBayessLearning()


# researchOrderer("HOG", "HEALTHY", 0, "Proby Etapu trzeciego/Uczacy/Negatywne/*")
# preprecessCorrectedFacePartsAndAngles(analysedData, 0, "Proby Etapu trzeciego/Uczacy/Wyjsciowe/")
# #
# researchOrderer("HOG", "HEALTHY", 0, "Proby Etapu trzeciego/Uczacy/Pozytywne/*")
# preprecessCorrectedFacePartsAndAngles(analysedData, 1, "Proby Etapu trzeciego/Uczacy/Wyjsciowe/")


# researchOrderer2("HOG", "HEALTHY", 0, "Proby Etapu trzeciego/Uczacy/Negatywne/*")
# preprecessCorrectedFacePartsAndAngles(analysedData, 0, "Proby Etapu trzeciego/Uczacy/Wyjsciowe/")

# researchOrderer2("HOG", "HEALTHY", 0, "Proby Etapu trzeciego/Uczacy/Pozytywne/*")
# preprecessCorrectedFacePartsAndAngles(analysedData, 1, "Proby Etapu trzeciego/Uczacy/Wyjsciowe/")

# researchOrderer("HOG", "HEALTHY", 0, "Proby Etapu trzeciego/Testowy/Negatywne/*")
# preprecessCorrectedFacePartsAndAngles(analysedData, 2, "Proby Etapu trzeciego/Uczacy/Wyjsciowe/")
# #
# #
# researchOrderer("HOG", "HEALTHY", 0, "Proby Etapu trzeciego/Testowy/Pozytywne/Ciezkie/*")
# preprecessCorrectedFacePartsAndAngles(analysedData, 3, "Proby Etapu trzeciego/Uczacy/Wyjsciowe/")
#
#


# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.809, 44], "Proby Etapu trzeciego/Testowy/Pozytywne/Lekkie/*")
# preprecessCorrectedFacePartsAndAngles(analysedData, 4, "Proby Etapu trzeciego/Uczacy/Wyjsciowe/")


researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.809, 44], 0)

file.close()

'''
0-5 porównanie części twarzy : 
fałdek nosowych, kącików ust, kącików oczu, dolnej powieki oraz przekroju ust
6-12 różne kąty pomiędzy interesującymi częściami twarzy z czego: 
(oczy,nos),(dolna część nosa,nos), (usta,nos),(lewarozwartosc ust),(prawa rozwartosc ust ),(lewy kacik,nos),(prawy kacik,nos)
13-91 punkty(x,y) opisujące części twarzy w kolejności : 
nos(4),dolna czesc nosa(5),lewe oko(6),prawe oko (6),usta(20)
ostatni punkt określa klasę obiektu (0,1)
'''
