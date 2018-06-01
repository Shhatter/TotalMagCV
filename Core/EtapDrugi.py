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
from skimage.measure import structural_similarity as ssim

from skimage import exposure
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
from dataStorage import DataStorage
from skimage import exposure
from skimage import feature
from skimage import img_as_ubyte

###STAŁE
predictor_path = "landmark/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# net = cv2.dnn.readNetFromCaffe("landmark/deploy.prototxt.txt", "landmark/res10_300x300_ssd_iter_140000.caffemodel")
mmod_path = "landmark/mmod_human_face_detector.dat"
cnnFaceDetector = dlib.cnn_face_detection_model_v1("landmark/mmod_human_face_detector.dat")

# Core/landmark/vgg_face_caffe/vgg_face_caffe/VGG_FACE_deploy.prototxt
# net = cv2.dnn.readNetFromCaffe("landmark/PAM_frontal_AlexNet/PAM_frontal_deploy.prototxt.txt", "landmark/PAM_frontal_AlexNet/snap__iter_100000.caffemodel")


faceFolderPath = "Pozytywne/*"
badFaceFolderPath = "Negatywne/"
positiveLister = glob.glob(faceFolderPath)
# HaarCascade prepare data
haarFaceCascade = cv2.CascadeClassifier('HaarCascadeConfigs/haarcascade_frontalface_default.xml')
lbpCascade = cv2.CascadeClassifier('HaarCascadeConfigs/lbpcascade_frontalface_improved.xml')
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
    inputFile = imutils.resize(inputFile, 1000)
    # ( Width [0], Height [1]
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    height, width = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")

    rects = detector(grayImage, 1)
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
            shape = predictor(grayImage, rect)
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

        roi_gray = grayImage[y:y + height, x:x + w]
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
        fa = FaceAligner(predictor, desiredFaceWidth=1000)
        faceOrig = imutils.resize(inputFile[y:y + h, x:x + w], width=256)
        xCenter = (int)((w / 2) + x)
        yCenter = (int)((h / 2) + y)
        lSideLength = math.sqrt(math.pow(fp48[0] - fp4[0], 2) + (math.pow(fp48[1] - fp4[1], 2)))
        rSideLength = math.sqrt(math.pow(fp54[0] - fp12[0], 2) + (math.pow(fp54[1] - fp12[1], 2)))
        if (((lSideLength * 0.666) > rSideLength) or ((rSideLength * 0.666) > lSideLength)):
            grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
            # for (x, y) in shape:
            #     cv2.circle(grayImage, (x, y), 1, (0, 0, 255), 5)

            # cv2.circle(grayImage, (xCenter, yCenter), 1, (200, 255, 23), 5)
            cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, grayImage)
        else:
            grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
            faceAligned = fa.align(inputFile, grayImage, rect)
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
            # cv2.imshow("Aligned", faceAligned)
            # cv2.waitKey(0)
            # counter += 1
            # print(counter)

            # cv2.circle(faceAligned, (xCenter, yCenter), 1, (200, 255, 23), 5)
            analysedData.append(DataStorage(inputFilePath, 0, inputFile, faceAligned, shape, False))
            print("analysedData: " + str(analysedData.__len__()))

            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, faceAligned)

            # cv2.imshow("Aligned", faceAligned)
            # cv2.waitKey(0)


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
    if (alghoritmName == "HOG"):
        if (mode == "SICK"):
            file.writelines("Positive\t")
            print("HOG: Sick People")

            pathCore = researchDefPath + "HOG\\" + getXTime + " HOG " + "\\"
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
            print("HOG: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + " HOG" + "\\"
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
                if printDetails:
                    printDetails = False
            printDetails = True
            falsePositive += badResult
            trueNegative += goodResult
            goodResult = 0
            badResult = 0
            ######################################################################################
            file.writelines("\n\npitch\t" + "\troll\t" "\tyaw\t" + "filename\n")
            for i in range(7, 9, 1):

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibFaceDetector(image, pathGood, pathGoodBad)
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
                    if printDetails:
                        printDetails = False
                printDetails = True
                falsePositive += badResult
                trueNegative += goodResult
                goodResult = 0
                badResult = 0
        # file.writelines(
        #     getTime + "\tHOG: " + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
        #     str(falseNegative) + "\tfalsePositive:\t" + str(
        #         falsePositive) + "\ttrueNegative:\t" + str(
        #         trueNegative) + "\tTotal:\t" + str(
        #         truePositive + trueNegative + falsePositive + falseNegative))


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
    return gray


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


def gen_sift_features(gray_img):
    sift = cv2.xfeatures2d.SIFT_create()
    # kp is the keypoints
    #
    # desc is the SIFT descriptors, they're 128-dimensional vectors
    # that we can use for our final features
    kp, desc = sift.detectAndCompute(gray_img, None)
    return kp, desc


def show_sift_features(gray_img, color_img, kp):
    cv2.imshow("nopper", cv2.drawKeypoints(gray_img, kp, color_img.copy()))
    cv2.waitKey(0)


def firstFaceDetector(aData):
    global getXTime
    getTimeFolderPersons = datetime.datetime.now()
    pathCore = personDefPath + getXTime + " ANALYZED DATA " + "\\"
    getXTime = str(getTimeFolderPersons.strftime("%Y-%m-%d - %H-%M-%S"))
    pathCore = pathCore.replace(":", " ")
    pathGood = pathCore + "Dobre\\"
    pathBad = pathCore + "Zle\\"
    os.mkdir(pathCore)
    os.mkdir(pathGood)
    os.mkdir(pathBad)

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

        leftNosePart, rightNosePartREV = resizeImagestToSameLevel(leftNosePart, rightNosePartREV)
        leftMounthEdge, rightMounthEdgeREV = resizeImagestToSameLevel(leftMounthEdge, rightMounthEdgeREV)
        leftEyeEdge, rightEyeEdgeREV = resizeImagestToSameLevel(leftEyeEdge, rightEyeEdgeREV)
        leftUnderEye, rightUnderEyeREV = resizeImagestToSameLevel(leftUnderEye, rightUnderEyeREV)
        mounthLeftPart, mounthRightPartREV = resizeImagestToSameLevel(mounthLeftPart, mounthRightPartREV)

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

        analysedParts = [[leftNosePartGRAY, rightNosePartREVGRAY], [leftMounthEdgeGRAY, rightMounthEdgeREVGRAY],
                         [leftEyeEdgeGRAY, rightEyeEdgeREVGRAY], [leftUnderEyeGRAY, rightUnderEyeREVGRAY],
                         [mounthLeftPartGRAY, mounthRightPartREVGRAY]]

        for singleComparasion in analysedParts:
            # singleComparasion[0] = cv2.equalizeHist(singleComparasion[0])
            # singleComparasion[1] = cv2.equalizeHist(singleComparasion[1])
            #
            aa = exposure.equalize_adapthist(singleComparasion[0], clip_limit=0.03)
            bb = exposure.equalize_adapthist(singleComparasion[1], clip_limit=0.03)
            # th1 = cv2.adaptiveThreshold(singleComparasion[0], 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            #                            cv2.THRESH_BINARY, 11, 2)
            # th2 = cv2.adaptiveThreshold(singleComparasion[1], 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            #                            cv2.THRESH_BINARY, 11, 2)
            res = np.hstack((singleComparasion[0], singleComparasion[1]))
            image = cv2.GaussianBlur(res, (5, 5), 0)

            singleComparasion[0]
            cv2.normalize()

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl1 = clahe.apply(res)
            cv2.imshow("argumented", cl1)
            cv2.imshow("normal", res)

            cv2.waitKey(0)

            aaa = cv2.Laplacian(res, cv2.CV_64F)
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
            sobelCombined = cv2.bitwise_or(sobelx, sobely)

            # cv2.imshow("arfg",aaa)
            # cv2.waitKey(0)
            # cv2.imshow("arfg",sobelx)
            # cv2.waitKey(0)
            # cv2.imshow("Blurred", image)
            # cv2.waitKey(0)
            #
            # cv2.imshow("arfg",sobelCombined)
            # cv2.waitKey(0)

            # edges1 = feature.canny(aa,sigma=2)
            # edges2 = feature.canny(bb,sigma=2)
            # # res2 = np.hstack((edges1, edges2))
            # cv_image1 = img_as_ubyte(edges1)
            # cv_image2 = img_as_ubyte(edges2)
            # res2 = np.hstack((cv_image1, cv_image2))
            # cv2.imshow("dupa",res2)
            # # cv2.imshow("arfg",res2)
            # cv2.waitKey(0)

            # fd, hogImage = hog(cleanImage, orientations=9, pixels_per_cell=(3, 3),
            #             #                    cells_per_block=(1, 1), visualise=True, feature_vector=True, transform_sqrt=True)
            #             #
            #             # fd2, hogImage2 = hog(singleComparasion[1], orientations=9, pixels_per_cell=(3, 3),
            #             #                      cells_per_block=(1, 1), visualise=True, feature_vector=True, transform_sqrt=True)

            # # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
            # #
            # # ax1.axis('off')
            # # ax1.imshow(singleComparasion[0], cmap=plt.cm.gray)
            # # ax1.set_title('Input image')
            # # hog_image_rescaled = exposure.rescale_intensity(hogImage, in_range=(0, 10))
            # # ax2.axis('off')
            # # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
            # # ax2.set_title('Histogram of Oriented Gradients')
            # a = []
            # for x in fd :
            #   a.append(int(180*x))
            # print("###############################################################")
            # # print(len(fd))
            # # print(len(fd2))
            # # plt.hist(fd)
            # # # plt.hist(fd, bins= 9, normed =True, alpha =0.5,histtype='stepfilled', color='steelblue',edgecolor='none')
            # # plt.show()
            # plt.hist(a)
            # plt.show()
            # cv2.imshow("clean image ", cleanImage)
            # cv2.waitKey(0)

        leftNosePartKP, leftNosePartDESC = gen_sift_features(leftNosePartGRAY)
        rightNosePartREVKP, rightNosePartREVDESC = gen_sift_features(rightNosePartREVGRAY)
        leftMounthEdgeKP, leftMounthEdgeDESC = gen_sift_features(leftMounthEdgeGRAY)
        rightMounthEdgeREVKP, rightMounthEdgeREVDESC = gen_sift_features(rightMounthEdgeREVGRAY)
        leftEyeEdgeKP, leftEyeEdgeDESC = gen_sift_features(leftEyeEdgeGRAY)
        rightEyeEdgeREVKP, rightEyeEdgeREVDESC = gen_sift_features(rightEyeEdgeREVGRAY)
        leftUnderEyeKP, leftUnderEyeDESC = gen_sift_features(leftUnderEyeGRAY)
        rightUnderEyeREVKP, rightUnderEyeREVDESC = gen_sift_features(rightUnderEyeREVGRAY)
        mounthLeftPartKP, mounthLeftPartDESC = gen_sift_features(mounthLeftPartGRAY)
        mounthRightPartREVKP, mounthRightPartREVDESC = gen_sift_features(mounthRightPartREVGRAY)

        # show_sift_features(leftNosePartGRAY, leftNosePart, leftNosePartKP)
        # show_sift_features(rightNosePartREVGRAY, rightNosePartREV, rightNosePartREVKP)
        # show_sift_features(leftMounthEdgeGRAY, leftMounthEdge, leftMounthEdgeKP)
        # show_sift_features(rightMounthEdgeREVGRAY, rightMounthEdgeREV, rightMounthEdgeREVKP)
        # show_sift_features(leftEyeEdgeGRAY, leftEyeEdge, leftEyeEdgeKP)
        # show_sift_features(rightEyeEdgeREVGRAY, rightEyeEdgeREV, rightEyeEdgeREVKP)
        # show_sift_features(leftUnderEyeGRAY, leftUnderEye, leftUnderEyeKP)
        # show_sift_features(rightUnderEyeREVGRAY, rightUnderEyeREV, rightUnderEyeREVKP)
        # show_sift_features(mounthLeftPartGRAY, mounthLeftPart, mounthLeftPartKP)
        # show_sift_features(mounthRightPartREVGRAY, mounthRightPartREV, mounthRightPartREVKP)

        # x1height, x1width = leftNosePart.shape[:2]
        # x2height, x2width = rightNosePartREV.shape[:2]
        # if (x1height * x1width > x2height * x2width):
        #     leftNosePart = imutils.resize(leftNosePart, x2width, x2height)
        # else:
        #     rightNosePartREV = imutils.resize(rightNosePartREV, x1width, x1height)
        #
        # x1height, x1width = leftMounthEdge.shape[:2]
        # x2height, x2width = rightMounthEdgeREV.shape[:2]
        # if (x1height * x1width > x2height * x2width):
        #     leftMounthEdge = imutils.resize(leftMounthEdge, x2width, x2height)
        # else:
        #     rightMounthEdgeREV = imutils.resize(rightMounthEdgeREV, x1width, x1height)
        #
        # x1height, x1width = leftEyeEdge.shape[:2]
        # x2height, x2width = rightEyeEdgeREV.shape[:2]
        # if (x1height * x1width > x2height * x2width):
        #     leftEyeEdge = imutils.resize(leftEyeEdge, x2width, x2height)
        # else:
        #     rightEyeEdgeREV = imutils.resize(rightEyeEdgeREV, x1width, x1height)
        #
        # x1height, x1width = leftEyeEdge.shape[:2]
        # x2height, x2width = rightEyeEdgeREV.shape[:2]
        # if (x1height * x1width > x2height * x2width):
        #     leftEyeEdge = imutils.resize(leftEyeEdge, x2width, x2height)
        # else:
        #     rightEyeEdgeREV = imutils.resize(rightEyeEdgeREV, x1width, x1height)
        #
        # x1height, x1width = leftUnderEye.shape[:2]
        # x2height, x2width = rightUnderEyeREV.shape[:2]
        # if (x1height * x1width > x2height * x2width):
        #     leftUnderEye = imutils.resize(leftUnderEye, x2width, x2height)
        # else:
        #     rightUnderEyeREV = imutils.resize(rightUnderEyeREV, x1width, x1height)
        #
        # x1height, x1width = mounthLeftPart.shape[:2]
        # x2height, x2width = mounthRightPartREV.shape[:2]
        # if (x1height * x1width > x2height * x2width):
        #     mounthLeftPart = imutils.resize(mounthLeftPart, x2width, x2height)
        # else:
        #     mounthRightPartREV = imutils.resize(mounthRightPartREV, x1width, x1height)

        # cv2.imshow("1",
        #            leftNosePart)
        # cv2.waitKey(0)
        # cv2.imshow("2",
        #            rightNosePartREV)
        # cv2.waitKey(0)
        # cv2.imshow("3",
        #            leftMounthEdge)
        # cv2.waitKey(0)
        # cv2.imshow("4",
        #            rightMounthEdgeREV)
        # cv2.waitKey(0)
        # cv2.imshow("5",
        #            leftEyeEdge)
        # cv2.waitKey(0)
        # cv2.imshow("6",
        #            rightEyeEdgeREV)
        # cv2.waitKey(0)
        # cv2.imshow("7",
        #            leftUnderEye)
        # cv2.waitKey(0)
        # cv2.imshow("8",
        #            rightUnderEyeREV)
        # cv2.waitKey(0)
        # cv2.imshow("9",
        #            mounthLeftPart)
        # cv2.waitKey(0)
        # cv2.imshow("10",
        #            mounthRightPartREV)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.imwrite(pathGood + pathlib.Path(member.fileName).name, member.alignedImage)


researchOrderer("HOG", "HEALTHY", 0, 0)
firstFaceDetector(analysedData)
file.close()
