import glob
import pathlib

import cmath
import dlib
import cv2
import imutils
import numpy as np
import shutil
import datetime
import tensorflow as tf
# OPEN CV
import os
from imutils import face_utils
from mtcnn.mtcnn import MTCNN
from mtcnn_facematch import detect_face
from time import sleep
import argparse

from skimage import io

sess = tf.Session()
# read pnet, rnet, onet models from align directory and files are det1.npy, det2.npy, det3.npy
pnet, rnet, onet = detect_face.create_mtcnn(sess, 'align')
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
###

### Sprawdzenie czy istnieje plik do logów
getTime = str(datetime.datetime.now().ctime())
if not (pathlib.Path("LogFile_Etap2.txt").is_file()):
    # os.mknod("/LogFile.txt",0)
    file = open("LogFile.txt", 'a')
    file.writelines(
        "\n##################################################################### " + "\nTest : " + getTime + "\n\n")
else:
    file = open("LogFile.txt", 'a')
    file.writelines(
        "\n##################################################################### " + "\nTest : " + getTime + "\n\n")


def getFace(inputFilePath, threshold, factor, goodPath, badPath):
    global goodResult, badResult
    inputFile = cv2.imread(inputFilePath)
    inputFile = imutils.resize(inputFile, width=1100)
    faces = []
    height, width = inputFile.shape[:2]
    bounding_boxes, _ = detect_face.detect_face(inputFile, int(width * 0.2), pnet, rnet, onet, threshold, factor)
    x = None
    y = None
    hPoint = None
    wPoint = None
    if (len(bounding_boxes) == 0):
        # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        # cv2.imshow("image", inputFile)
        # cv2.waitKey(0)
        badResult += 1

    elif (len(bounding_boxes) == 1):

        if (bounding_boxes[0][4] > 0.70):
            print("highest: " + str(bounding_boxes[0][4]))

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

            cv2.rectangle(inputFile, (x, y),
                          (wPoint, hPoint), (0, 255, 0), 2)
            goodResult += 1
            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)
        else:
            # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
            badResult += 1

    elif (len(bounding_boxes) != 1):

        highest = bounding_boxes[0]
        for i in range(1, len(bounding_boxes), 1):
            print("highest: " + str(highest[0]))

            if (bounding_boxes[i][4] > highest[4]):
                highest = bounding_boxes[i]

        print("highest: " + str(highest[0]))

        if (highest[4] > 0.70):
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

            cv2.rectangle(inputFile, (x, y),
                          (wPoint, hPoint), (0, 255, 0), 2)
            goodResult += 1

            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)
        else:
            # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
            badResult += 1


def haarCascadeFaceDetector(inputFilePath, scaleFactor, neighbours, goodPath, badPath):
    # if printDetails:
    #     file.writelines(
    #         getTime + "\t" + "Haar Cascade: neighbours:\t" + str(neighbours) + "\tscaleFactor:\t" + str(
    #             scaleFactor) + "\t")
    #     file.writelines("haarFaceCascade" + "\n")
    #     file.writelines("scaleFactor: " + str(scaleFactor) + "\nneighbours: " + str(neighbours) + "\n\n")

    inputFile = cv2.imread(inputFilePath)
    width, height = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = haarFaceCascade.detectMultiScale(grayImage, scaleFactor, neighbours)
    global goodResult, badResult

    if len(detectedFace) != 1:
        # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        print(len(detectedFace))
        badResult += 1
    else:
        goodResult += 1
        for x, y, w, h in detectedFace:
            # Pokazanie że wykrywa twarz - można pominąć
            cv2.rectangle(inputFile, (x, y), (x + w, y + int(h + (h * 0.2))), (255, 0, 0), 2)
            smart_h = int(h * chinHeightROI)
            if smart_h > height:
                roi_color = inputFile[y:y + (height - 1), x:x + w]
            else:
                roi_color = inputFile[y:y + h + int(smart_h), x:x + w]

            roi_gray = grayImage[y:y + h, x:x + w]
            # croppedImage = cv2.clone
            # cv2.imwrite('WynikiAnalizy\\Haar Cascade\\Dobre\\' + pathlib.Path(inputFilePath).name, roi_color)
            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)

        # cv2.imshow("image",roi_color)
        # cv2.waitKey(0)

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

    #
    # cv2.imshow("Image", inputFile)
    # cv2.waitKey(0)


def lbpCascadeDetector(inputFilePath, scaleFactor, neighbours, goodPath, badPath):
    # if printDetails:
    #     file.writelines(
    #         getTime + "\t" + "LBP: neighbours:\t" + str(neighbours) + "\tscaleFactor:\t" + str(scaleFactor) + "\t")
    # #     file.writelines("lbpCascadeDetector" + "\n\n")
    #     file.writelines("scaleFactor: " + str(scaleFactor) + "\nneighbours: " + str(neighbours) + "\n\n")

    global goodResult, badResult
    inputFile = cv2.imread(inputFilePath)

    width, height = inputFile.shape[:2]
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    detectedFace = lbpCascade.detectMultiScale(grayImage, scaleFactor, neighbours)

    if len(detectedFace) != 1:
        # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
        badResult += 1
    else:
        goodResult += 1
        for x, y, w, h in detectedFace:
            smart_h = int(h * chinHeightROI)
            if smart_h > height:
                roi_color = inputFile[y:y + (height - 1), x:x + w]
            else:
                roi_color = inputFile[y:y + h + int(smart_h), x:x + w]
            # Pokazanie że wykrywa twarz - można pominąć

            cv2.rectangle(inputFile, (x, y), (x + w, y + int(h + (h * 0.2))), (255, 0, 0), 2)
            roi_gray = grayImage[y:y + h, x:x + w]
            # croppedImage = cv2.clone
            # cv2.imwrite('WynikiAnalizy\\LBP\\Dobre\\' + pathlib.Path(inputFilePath).name, roi_color)
            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)

        # cv2.imshow("image",roi_color)
        # cv2.waitKey(0)

    # if (detectedFace):
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # for (x,y,w,h) in detectedFace:
    #     cv2.rectangle(dSave,(x,y),(x+w,y+h),(255,0,0),2)

    #
    # cv2.imshow("Image", inputFile)
    # cv2.waitKey(0)


def dlibFaceDetector(inputFilePath, resize, goodPath, badPath):
    # if printDetails:
    # file.writelines(
    #     getTime + "\t" + "Histogram of Oriented Gradients: (neighbours:\t")
    global badResult, goodResult
    inputFile = cv2.imread(inputFilePath)

    inputFile = imutils.resize(inputFile, resize)
    # ( Width [0], Height [1]
    # inputFile = imutils.resize(inputFile, 500)
    grayImage = cv2.cvtColor(inputFile, cv2.COLOR_BGR2GRAY)
    width, height = inputFile.shape[:2]
    print("width: " + str(width) + " height: " + str(height) + "\n")
    rects = detector(grayImage, 1)
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

            cv2.rectangle(inputFile, (x, y), (x + w, y + h), (0, 255, 0), 2)

            smart_h = int(h * chinHeightROI)
            roi_color = inputFile[y:y + h, x:x + w]
            #
            # if smart_h > h:
            #     roi_color = inputFile[y:y + (height - 1), x:x + w]
            # else:
            #     roi_color = inputFile[y:y + h + int(smart_h), x:x + w]

            roi_gray = grayImage[y:y + height, x:x + w]
            # croppedImage = cv2.clone
            # cv2.imshow("Output", roi_color)
            # cv2.waitKey(0)
            # show the face number
            # cv2.putText(inputFile, "Face #{}".format(i + 1), (x - 10, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(inputFile, (x, y), 1, (0, 0, 255), -1)

            # show the output image with the face detections + facial landmarks
            # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)


def mtcnnDetector(inputFilePath, goodPath, badPath):
    # if printDetails:
    #     file.writelines(
    #         getTime + "\t" + "Histogram of Oriented Gradients: (neighbours:\t")
    global badResult, goodResult
    inputFile = cv2.imread(inputFilePath)

    detector = MTCNN()
    result = detector.detect_faces(inputFile)
    bounding_box = result[0]['box']

    keypoints = result[0]['keypoints']

    cv2.rectangle(inputFile,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0, 155, 255),
                  2)
    cv2.imshow("Output", inputFile)
    cv2.waitKey(0)


# def caffeDeepLearningDetector(inputFilePath, globalConf, resizeSize, goodPath, badPath):
#     # file.writelines(
#     #     getTime + "\t" + "DEEPLEARNING_CAFFE: globalConf:\t" + str(globalConf) + "\tresizeSize:\t" + str(
#     #         resizeSize) + "\t")
#     global badResult, goodResult
#     inputFile = cv2.imread(inputFilePath)
#     (h, w) = inputFile.shape[:2]
#     blob = cv2.dnn.blobFromImage(cv2.resize(inputFile, (resizeSize, resizeSize)), 1.0,
#                                  (300, 300), (104.0, 177.0, 123.0))
#     # inputFile = imutils.resize(inputFile, resizeSize)
#     blob = cv2.dnn.blobFromImage(inputFile)
#     net.setInput(blob)
#     detections = net.forward()
#     counter = 0
#     for i in range(0, detections.shape[2]):
#
#         confidence = detections[0, 0, i, 2]
#         if confidence > globalConf:
#             counter += 1
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#
#             text = "{:.2f}%".format(confidence * 100)
#             y = startY - 10 if startY - 10 > 10 else startY + 10
#             cv2.rectangle(inputFile, (startX, startY), (endX, endY),
#                           (0, 0, 255), 2)
#             cv2.putText(inputFile, text, (startX, y),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
#             # cv2.imshow("Output", inputFile)
#             # cv2.waitKey(0)
#     if counter != 1:
#         cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, inputFile)
#         print(counter)
#         badResult += 1
#     else:
#         cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, inputFile)
#         print(counter)
#         goodResult += 1


def dlibDeepLearningDetector(inputFilePath, resizeSize, upsample, goodPath, badPath):
    # dlib.cuda.buffer
    # if printDetails:
    #     file.writelines(
    #         getTime + "\t" + "DEEPLEARNING_CNN: resizeSize:\t" + str(resizeSize) + "\tupsample:\t" + str(
    #             upsample) + "\t")
    # win = dlib.image_window()
    global badResult, goodResult
    inputFile = cv2.imread(inputFilePath)
    width, height = inputFile.shape[:2]
    if (resizeSize > 0):
        inputFile = cv2.resize(inputFile, (resizeSize, resizeSize))

    copyColor = inputFile

    dets = cnnFaceDetector(inputFile, upsample)
    print("Number of faces detected: {}".format(len(dets)))

    if (len(dets) != 1):
        # cv2.imwrite(badPath + pathlib.Path(inputFilePath).name, copyColor)
        badResult += 1
    else:
        goodResult += 1

        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))

        # rects = dlib.rectangles()
        (x, y, w, h) = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
        # if x < 0:
        #     x = 0
        # elif x > width:
        #     x = width - 1
        #
        # if (y < 0):
        #     y = 0
        # elif y > height:
        #     y = height - 1
        #
        # if w < 0:
        #     w = 0
        # elif w > width:
        #     w = width - 1
        #
        # if (h < 0):
        #     h = 0
        # elif h > height:
        #     h = height - 1
        cv2.rectangle(copyColor, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.imshow("Output", copyColor)
        # cv2.waitKey(0)
        # cv2.imwrite(goodPath + pathlib.Path(inputFilePath).name, copyColor)

    # rects.extend([d.rect for d in dets])
    #
    # win.clear_overlay()
    # win.set_image(inputFile)
    # win.add_overlay(rects)
    # dlib.hit_enter_to_continue()


def f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, beta):
    print("working hard")

    accuracy = (truePositive + trueNegative) / (truePositive + trueNegative + falsePositive + falseNegative)
    precision = truePositive / (truePositive + falsePositive)
    recall = truePositive / (truePositive + falseNegative)
    fallout = falsePositive / (falsePositive + trueNegative)
    fMeasure = (1 + pow(beta, 2)) * ((precision * recall) / (pow(beta, 2) * (precision + recall)))
    return "\taccuracy:\t" + str(accuracy) + "\tprecision:\t" + str(precision) + "\trecall(TPR):\t" + str(
        recall) + "\tfallout(FPR):\t" + str(fallout) + "\tfMeasure:\t" + str(fMeasure)


getTimeFolderPersons = datetime.datetime.now()
getXTime = str(getTimeFolderPersons.strftime("%Y-%m-%d - %H-%M-%S"))


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

    if (alghoritmName == "HAAR"):

        if (mode == "SICK"):
            print("HAAR: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + "SICK Haar SF " + str(values[0]) + " NB " + str(
                values[1]) + "\\"
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

            for i in range(11, 12, 1):
                # pathCore = personDefPath + str(i) + "\\" + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                #     values[1]) + "\\"

                # os.mkdir(pathCore)

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                truePositive += goodResult
                falseNegative += badResult
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Good_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    haarCascadeFaceDetector(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                # file.writelines("Results:\t")
                # file.writelines("Bad:\t" + str(goodResult) + '\t')
                # file.writelines("Bad_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0
            file.writelines(
                getTime + "\tSICK\tHaar Cascade: scaleFactor:_" + str(values[0]) + "_neighbours:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")
        elif (mode == "HEALTHY"):
            print("HAAR: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + "HEALTHY Haar SF " + str(values[0]) + " NB " + str(
                values[1]) + "\\"
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

            for i in range(1, 11, 1):
                # pathCore = personDefPath + str(i) + "\\" + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                #     values[1]) + "\\"

                # os.mkdir(pathCore)

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                truePositive += goodResult
                falseNegative += badResult
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Good_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    haarCascadeFaceDetector(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                # file.writelines("Results:\t")
                # file.writelines("Bad:\t" + str(goodResult) + '\t')
                # file.writelines("Bad_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

            file.writelines(
                getTime + "\tHEALTHY\tHaar Cascade: scaleFactor:_" + str(values[0]) + "_neighbours:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")

    elif (alghoritmName == "LBP"):
        if (mode == "SICK"):
            print("LBP: SICK People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + "SICK LBP SF " + str(values[0]) + " NB " + str(
                values[1]) + "\\"
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

            for i in range(11, 12, 1):
                # pathCore = personDefPath + str(i) + "\\" + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                #     values[1]) + "\\"

                # os.mkdir(pathCore)

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    lbpCascadeDetector(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                truePositive += goodResult
                falseNegative += badResult
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Good_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    lbpCascadeDetector(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                # file.writelines("Results:\t")
                # file.writelines("Bad:\t" + str(goodResult) + '\t')
                # file.writelines("Bad_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

            file.writelines(
                getTime + "\tSICK\tLBP :CNNscaleFactor:" + str(values[0]) + "_neighbours:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")
        elif (mode == "HEALTHY"):
            print("LBP: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + "HEALTHY LBP SF " + str(values[0]) + " NB " + str(
                values[1]) + "\\"
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

            for i in range(1, 11, 1):
                # pathCore = personDefPath + str(i) + "\\" + getXTime + " Haar SF " + str(values[0]) + " NB " + str(
                #     values[1]) + "\\"

                # os.mkdir(pathCore)

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    lbpCascadeDetector(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                truePositive += goodResult
                falseNegative += badResult
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Good_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     haarCascadeFaceDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    lbpCascadeDetector(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                # file.writelines("Results:\t")
                # file.writelines("Bad:\t" + str(goodResult) + '\t')
                # file.writelines("Bad_failed:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                goodResult = 0
                badResult = 0

            file.writelines(
                getTime + "\tHEALTHY\tLBP :CNNscaleFactor:" + str(values[0]) + "_neighbours:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")

    elif (alghoritmName == "CNNDLIB"):
        if (mode == "SICK"):
            print("CNNDLIB: Sick People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + "SICK CNNDLIB RESIZE" + str(
                values[0]) + " UPSAMPLE " + str(
                values[1]) + "\\"
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
                    dlibDeepLearningDetector(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                truePositive += goodResult
                falseNegative += badResult
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     dlibDeepLearningDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibDeepLearningDetector(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                falsePositive += badResult
                trueNegative += goodResult
                goodResult = 0
                badResult = 0
            file.writelines(
                getTime + "\tSICK\tDEEPLEARNING_CNN :resizeSize:" + str(values[0]) + "_upsample:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")

        elif (mode == "HEALTHY"):
            print("CNNDLIB: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + "HEALTHY CNNDLIB RESIZE" + str(
                values[0]) + " UPSAMPLE " + str(
                values[1]) + "\\"
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
            for i in range(1, 11, 1):

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibDeepLearningDetector(image, values[0], values[1], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                truePositive += goodResult
                falseNegative += badResult
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                # counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     dlibDeepLearningDetector(image, values[0], values[1], pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibDeepLearningDetector(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                falsePositive += badResult
                trueNegative += goodResult
                goodResult = 0
                badResult = 0
            file.writelines(
                getTime + "\tHEALTHY\tDEEPLEARNING_CNN :resizeSize:" + str(values[0]) + "_upsample:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")

    elif (alghoritmName == "HOG"):

        if (mode == "SICK"):
            print("HOG: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + "SICK HOG " + "\\"
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
                    dlibFaceDetector(image, values[0], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                truePositive += goodResult
                falseNegative += badResult
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                counter = 0
                # for image in lister_moderate:
                #     print(image)
                #     print("Iteracja: " + str(counter))
                #     counter += 1
                #     dlibFaceDetector(image, pathGood, pathBad)
                #     if printDetails:
                #         printDetails = False
                # printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                # goodResult = 0
                # badResult = 0
                #
                # file.writelines("Osoba " + str(i) + " " + "Zle" + ":\t")
                # counter = 0
                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibFaceDetector(image, values[0], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                # file.writelines("Results:\t")
                # file.writelines("Good:\t" + str(goodResult) + '\t')
                # file.writelines("Bad:\t" + str(badResult) + '\t')
                # file.writelines("Total:\t" + str(badResult + goodResult) + "\t\n")
                falsePositive += badResult
                trueNegative += goodResult
                goodResult = 0
                badResult = 0
            file.writelines(
                getTime + "\tSICK \tHOG: " + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")
        elif (mode == "HEALTHY"):
            print("HOG: Healthy People")
            # if clear == 0 :
            #     removeAllResults(00)
            pathCore = personDefPath + getXTime + "HEALTHY HOG " + "\\"
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
            for i in range(1, 11, 1):

                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                # lister_moderate = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Srednie/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibFaceDetector(image, values[0], pathGood, pathGoodBad)
                    if printDetails:
                        printDetails = False
                printDetails = True
                truePositive += goodResult
                falseNegative += badResult
                goodResult = 0
                badResult = 0

                # file.writelines("Osoba " + str(i) + " " + "Srednie" + ":\t")
                counter = 0

                for image in lister_bad:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
                    dlibFaceDetector(image, values[0], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                goodResult = 0
                badResult = 0
            file.writelines(
                getTime + "\tHEALTHY\tHOG: " + "\ttruePositive:\t" + str(truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")

    elif (alghoritmName == "MTCNN"):
        if (mode == "SICK"):
            print("MTCNN: Sick People")
            pathCore = personDefPath + getXTime + "SICK MTCNN: threshold:_" + str(
                values[0]) + "_factor:_" + str(values[1]) + "\\"
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

            for i in range(11, 12, 1):
                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
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
                    getFace(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                goodResult = 0
                badResult = 0

            file.writelines(
                getTime + "\tSICK\tMTCNN: threshold:_" + str(values[0]) + "_factor:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(
                    truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")
        elif (mode == "HEALTHY"):
            print("MTCNN: Healthy People")
            pathCore = personDefPath + getXTime + " MTCNN: threshold:_" + str(
                values[0]) + "_factor:_" + str(values[1]) + "\\"
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

            for i in range(1, 11, 1):
                lister_good = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Dobre/*")
                lister_bad = glob.glob("ProbkiBadawcze/Osoba" + str(i) + "/Zle/*")

                # file.writelines("Osoba " + str(i) + " " + "Dobre" + ":\t")
                counter = 0
                for image in lister_good:
                    print(image)
                    print("Iteracja: " + str(counter))
                    counter += 1
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
                    getFace(image, values[0], values[1], pathBadBad, pathBad)
                    if printDetails:
                        printDetails = False
                printDetails = True

                falsePositive += badResult
                trueNegative += goodResult
                goodResult = 0
                badResult = 0

            file.writelines(
                getTime + "\tHEALTHY\tMTCNN: threshold:_" + str(values[0]) + "_factor:_" + str(
                    values[1]) + "\ttruePositive:\t" + str(
                    truePositive) + "\tfalseNegative:\t" +
                str(falseNegative) + "\tfalsePositive:\t" + str(
                    falsePositive) + "\ttrueNegative:\t" + str(
                    trueNegative) + "\tTotal:\t" + str(
                    truePositive + trueNegative + falsePositive + falseNegative) +
                f1ScoreComputer(truePositive, trueNegative, falsePositive, falseNegative, 1) + "\n")


######################### B
######################### A
######################### D
######################### A
######################### N
######################### I
######################### A
# #
# # researchOrderer("CNNDLIB", "HEALTHY", [300, 1], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [200, 1], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [700, 1], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [800, 1], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [300, 2], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [200, 2], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [700, 2], 0)


# # file.writelines("Haar:\n")

# # researchOrderer("HAAR", "SICK", [2, 3], 0)
# # researchOrderer("HAAR", "SICK", [7, 8], 0)
# # researchOrderer("HAAR", "SICK", [3, 5], 0)
# # researchOrderer("HAAR", "SICK", [4, 10], 0)
# # researchOrderer("HAAR", "SICK", [5, 8], 0)
#
# researchOrderer("HAAR", "HEALTHY", [2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7, 8], 0)
# researchOrderer("HAAR", "HEALTHY", [3, 5], 0)
# researchOrderer("HAAR", "HEALTHY", [4, 10], 0)
# researchOrderer("HAAR", "HEALTHY", [5, 8], 0)
# # #
# #
# # file.writelines("LBP:\n")
# # researchOrderer("LBP", "SICK", [2, 3], 0)
# # researchOrderer("LBP", "SICK", [7, 8], 0)
# # researchOrderer("LBP", "SICK", [3, 5], 0)
# # researchOrderer("LBP", "SICK", [4, 10], 0)
# # researchOrderer("LBP", "SICK", [5, 8], 0)
# #
# researchOrderer("LBP", "HEALTHY", [2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7, 8], 0)
# researchOrderer("LBP", "HEALTHY", [3, 5], 0)
# researchOrderer("LBP", "HEALTHY", [4, 10], 0)
# researchOrderer("LBP", "HEALTHY", [5, 8], 0)
#
# # file.writelines("DEEPLEARNING_CAFFE:\n")
# researchOrderer("DLCAFFE", "HEALTHY", [0.5, 100], 0)
#
# researchOrderer("DLCAFFE", "HEALTHY", [0.5, 500], 0)
# researchOrderer("DLCAFFE", "HEALTHY", [0.5, 400], 0)
# researchOrderer("DLCAFFE", "HEALTHY", [0.5, 300], 0)
#
# researchOrderer("DLCAFFE", "HEALTHY", [0.5, 200], 0)


# researchOrderer("DLCAFFE", "SICK", [0.5, 300], 0)
#
# # researchModeExecutor(2, 2, 0, positiveLister, dlibGoodPath, dlibBadPath)
#
#
# researchOrderer("CNNDLIB", "SICK", [300, 1], 0)
# # researchOrderer("CNNDLIB", "SICK", [200, 1], 0)
# # researchOrderer("CNNDLIB", "SICK", [700, 1], 0)
# # researchOrderer("CNNDLIB", "SICK", [800, 1], 0)D
# researchOrderer("CNNDLIB", "HEALTHY", [0, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [0, 2], 0)

#
# # researchOrderer("CNNDLIB", "SICK", [300, 2], 0)
# # researchOrderer("CNNDLIB", "SICK", [200, 2], 0)
# # researchOrderer("CNNDLIB", "SICK", [700, 2], 0)
# # researchOrderer("CNNDLIB", "SICK", [800, 2], 0)
# # researchOrderer("CNNDLIB", "SICK", [0, 2], 0)
#
# print("druga seria")
# # # ZROBIONE
#
# # researchOrderer("CNNDLIB", "HEALTHY", [0, 2], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [800, 2], 0)
# # researchOrderer("CNNDLIB", "HEALTHY", [0, 1], 0)
#
# researchOrderer("HOG", "HEALTHY", 0, 0)
# researchOrderer("HOG", "SICK", 0, 0)

# researchOrderer("MTCNN","HEALTHY",0,0)


# researchOrderer("HAAR", "HEALTHY", [2, 4], 0)
# researchOrderer("HAAR", "HEALTHY", [7, 5], 0)
# researchOrderer("HAAR", "HEALTHY", [3, 5], 0)
# researchOrderer("HAAR", "HEALTHY", [4, 10], 0)
# researchOrderer("HAAR", "HEALTHY", [5, 8], 0)

# researchOrderer("LBP", "HEALTHY", [2, 4], 0)
# researchOrderer("LBP", "HEALTHY", [7, 8], 0)
# researchOrderer("LBP", "HEALTHY", [3, 5], 0)
# researchOrderer("LBP", "HEALTHY", [4, 10], 0)
# researchOrderer("LBP", "HEALTHY", [5, 8], 0)


# DODATKOWE BY WYRÓWNAC
# researchOrderer("CNNDLIB", "HEALTHY", [250, 3], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [250, 3], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [350, 2], 0)

# researchOrderer("LBP", "HEALTHY", [4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5, 4], 0)
# researchOrderer("LBP", "HEALTHY", [5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6, 3], 0)
#
# researchOrderer("HAAR", "HEALTHY", [4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5, 4], 0)
# researchOrderer("HAAR", "HEALTHY", [5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6, 3], 0)


# researchOrderer("HAAR", "SICK", [5, 8], 0)
# researchOrderer("LBP", "SICK", [5, 8], 0)
# researchOrderer("CNNDLIB", "SICK", [350, 2], 0)
# researchOrderer("HOG", "SICK", 0, 0)
# researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.809], 0)


# researchOrderer("HAAR", "HEALTHY", [5, 8], 0)
# researchOrderer("LBP", "HEALTHY", [5, 8], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [350, 2], 0)
# researchOrderer("HOG", "HEALTHY", 0, 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.809], 0)


# researchOrderer("HAAR", "HEALTHY", [1.1, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [1.2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [1.3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [1.4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [1.5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [1.6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [1.7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [1.8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [1.9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [2.1, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [2.2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [2.3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [2.4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [2.5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [2.6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [2.7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [2.8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [2.9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3.1, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3.2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3.3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3.4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3.5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3.6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3.7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3.8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [3.9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [4.1, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [4.2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [4.3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [4.4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [4.5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [4.6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [4.7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [4.8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [4.9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5.1, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5.2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5.3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5.4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5.5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5.6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5.7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5.8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [5.9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6.1, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6.2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6.3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6.4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6.5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6.6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6.7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6.8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [6.9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7.1, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7.2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7.3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7.4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7.5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7.6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7.7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7.8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [7.9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [8.1, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [8.2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [8.3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [8.4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [8.5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [8.6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [8.7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [8.8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [8.9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [9.1, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [9.2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [9.3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [9.4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [9.5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [9.6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [9.7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [9.8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [9.9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [10, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [10.1, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [10.2, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [10.3, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [10.4, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [10.5, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [10.6, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [10.7, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [10.8, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [10.9, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [11, 3], 0)
# researchOrderer("HAAR", "HEALTHY", [11.1, 3], 0)

# researchOrderer("LBP", "HEALTHY", [1.1, 3], 0)
# researchOrderer("LBP", "HEALTHY", [1.2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [1.3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [1.4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [1.5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [1.6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [1.7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [1.8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [1.9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [2.1, 3], 0)
# researchOrderer("LBP", "HEALTHY", [2.2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [2.3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [2.4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [2.5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [2.6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [2.7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [2.8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [2.9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3.1, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3.2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3.3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3.4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3.5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3.6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3.7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3.8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [3.9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [4.1, 3], 0)
# researchOrderer("LBP", "HEALTHY", [4.2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [4.3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [4.4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [4.5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [4.6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [4.7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [4.8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [4.9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5.1, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5.2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5.3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5.4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5.5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5.6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5.7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5.8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [5.9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6.1, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6.2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6.3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6.4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6.5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6.6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6.7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6.8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [6.9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7.1, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7.2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7.3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7.4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7.5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7.6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7.7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7.8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [7.9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [8.1, 3], 0)
# researchOrderer("LBP", "HEALTHY", [8.2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [8.3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [8.4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [8.5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [8.6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [8.7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [8.8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [8.9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [9.1, 3], 0)
# researchOrderer("LBP", "HEALTHY", [9.2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [9.3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [9.4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [9.5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [9.6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [9.7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [9.8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [9.9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [10, 3], 0)
# researchOrderer("LBP", "HEALTHY", [10.1, 3], 0)
# researchOrderer("LBP", "HEALTHY", [10.2, 3], 0)
# researchOrderer("LBP", "HEALTHY", [10.3, 3], 0)
# researchOrderer("LBP", "HEALTHY", [10.4, 3], 0)
# researchOrderer("LBP", "HEALTHY", [10.5, 3], 0)
# researchOrderer("LBP", "HEALTHY", [10.6, 3], 0)
# researchOrderer("LBP", "HEALTHY", [10.7, 3], 0)
# researchOrderer("LBP", "HEALTHY", [10.8, 3], 0)
# researchOrderer("LBP", "HEALTHY", [10.9, 3], 0)
# researchOrderer("LBP", "HEALTHY", [11, 3], 0)
# researchOrderer("LBP", "HEALTHY", [11.1, 3], 0)


# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.01], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.02], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.03], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.04], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.05], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.06], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.07], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.08], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.09], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.1], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.11], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.12], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.13], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.14], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.15], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.16], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.17], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.18], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.19], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.2], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.21], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.22], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.23], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.24], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.25], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.26], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.27], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.28], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.29], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.3], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.31], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.32], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.33], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.34], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.35], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.36], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.37], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.38], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.39], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.4], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.41], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.42], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.43], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.44], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.45], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.46], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.47], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.48], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.49], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.5], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.51], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.52], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.53], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.54], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.55], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.56], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.57], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.58], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.59], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.6], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.61], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.62], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.63], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.64], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.65], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.66], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.67], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.68], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.69], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.7], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.71], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.72], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.73], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.74], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.75], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.76], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.77], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.78], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.79], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.8], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.81], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.82], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.83], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.84], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.85], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.86], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.87], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.88], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.89], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.9], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.91], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.92], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.93], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.94], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.95], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.96], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.97], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.98], 0)
# researchOrderer("MTCNN", "HEALTHY", [[0.2, 0.5, 0.8], 0.99], 0)


# researchOrderer("CNNDLIB", "HEALTHY", [700, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [695, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [690, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [685, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [680, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [675, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [670, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [665, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [660, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [655, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [650, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [645, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [640, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [635, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [630, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [625, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [620, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [615, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [610, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [605, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [600, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [595, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [590, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [585, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [580, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [575, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [570, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [565, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [560, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [555, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [550, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [545, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [540, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [535, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [530, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [525, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [520, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [515, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [510, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [505, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [500, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [495, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [490, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [485, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [480, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [475, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [470, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [465, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [460, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [455, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [450, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [445, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [440, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [435, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [430, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [425, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [420, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [415, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [410, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [405, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [400, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [395, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [390, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [385, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [380, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [375, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [370, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [365, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [360, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [355, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [350, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [345, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [340, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [335, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [330, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [325, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [320, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [315, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [310, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [305, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [300, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [295, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [290, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [285, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [280, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [275, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [270, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [265, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [260, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [255, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [250, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [245, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [240, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [235, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [230, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [225, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [220, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [215, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [210, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [205, 1], 0)
# researchOrderer("CNNDLIB", "HEALTHY", [200, 1], 0)


# researchOrderer("HOG", "HEALTHY", [700], 0)
# researchOrderer("HOG", "HEALTHY", [695], 0)
# researchOrderer("HOG", "HEALTHY", [690], 0)
# researchOrderer("HOG", "HEALTHY", [685], 0)
# researchOrderer("HOG", "HEALTHY", [680], 0)
# researchOrderer("HOG", "HEALTHY", [675], 0)
# researchOrderer("HOG", "HEALTHY", [670], 0)
# researchOrderer("HOG", "HEALTHY", [665], 0)
# researchOrderer("HOG", "HEALTHY", [660], 0)
# researchOrderer("HOG", "HEALTHY", [655], 0)
# researchOrderer("HOG", "HEALTHY", [650], 0)
# researchOrderer("HOG", "HEALTHY", [645], 0)
# researchOrderer("HOG", "HEALTHY", [640], 0)
# researchOrderer("HOG", "HEALTHY", [635], 0)
# researchOrderer("HOG", "HEALTHY", [630], 0)
# researchOrderer("HOG", "HEALTHY", [625], 0)
# researchOrderer("HOG", "HEALTHY", [620], 0)
# researchOrderer("HOG", "HEALTHY", [615], 0)
# researchOrderer("HOG", "HEALTHY", [610], 0)
# researchOrderer("HOG", "HEALTHY", [605], 0)
# researchOrderer("HOG", "HEALTHY", [600], 0)
# researchOrderer("HOG", "HEALTHY", [595], 0)
# researchOrderer("HOG", "HEALTHY", [590], 0)
# researchOrderer("HOG", "HEALTHY", [585], 0)
# researchOrderer("HOG", "HEALTHY", [580], 0)
# researchOrderer("HOG", "HEALTHY", [575], 0)
# researchOrderer("HOG", "HEALTHY", [570], 0)
# researchOrderer("HOG", "HEALTHY", [565], 0)
# researchOrderer("HOG", "HEALTHY", [560], 0)
# researchOrderer("HOG", "HEALTHY", [555], 0)
# researchOrderer("HOG", "HEALTHY", [550], 0)
# researchOrderer("HOG", "HEALTHY", [545], 0)
# researchOrderer("HOG", "HEALTHY", [540], 0)
# researchOrderer("HOG", "HEALTHY", [535], 0)
# researchOrderer("HOG", "HEALTHY", [530], 0)
# researchOrderer("HOG", "HEALTHY", [525], 0)
# researchOrderer("HOG", "HEALTHY", [520], 0)
# researchOrderer("HOG", "HEALTHY", [515], 0)
# researchOrderer("HOG", "HEALTHY", [510], 0)
# researchOrderer("HOG", "HEALTHY", [505], 0)
# researchOrderer("HOG", "HEALTHY", [500], 0)
# researchOrderer("HOG", "HEALTHY", [495], 0)
# researchOrderer("HOG", "HEALTHY", [490], 0)
# researchOrderer("HOG", "HEALTHY", [485], 0)
# researchOrderer("HOG", "HEALTHY", [480], 0)
# researchOrderer("HOG", "HEALTHY", [475], 0)
# researchOrderer("HOG", "HEALTHY", [470], 0)
# researchOrderer("HOG", "HEALTHY", [465], 0)
# researchOrderer("HOG", "HEALTHY", [460], 0)
# researchOrderer("HOG", "HEALTHY", [455], 0)
# researchOrderer("HOG", "HEALTHY", [450], 0)
# researchOrderer("HOG", "HEALTHY", [445], 0)
# researchOrderer("HOG", "HEALTHY", [440], 0)
# researchOrderer("HOG", "HEALTHY", [435], 0)
# researchOrderer("HOG", "HEALTHY", [430], 0)
# researchOrderer("HOG", "HEALTHY", [425], 0)
# researchOrderer("HOG", "HEALTHY", [420], 0)
# researchOrderer("HOG", "HEALTHY", [415], 0)
# researchOrderer("HOG", "HEALTHY", [410], 0)
# researchOrderer("HOG", "HEALTHY", [405], 0)
# researchOrderer("HOG", "HEALTHY", [400], 0)
# researchOrderer("HOG", "HEALTHY", [395], 0)
# researchOrderer("HOG", "HEALTHY", [390], 0)
# researchOrderer("HOG", "HEALTHY", [385], 0)
# researchOrderer("HOG", "HEALTHY", [380], 0)
# researchOrderer("HOG", "HEALTHY", [375], 0)
# researchOrderer("HOG", "HEALTHY", [370], 0)
# researchOrderer("HOG", "HEALTHY", [365], 0)
# researchOrderer("HOG", "HEALTHY", [360], 0)
# researchOrderer("HOG", "HEALTHY", [355], 0)
# researchOrderer("HOG", "HEALTHY", [350], 0)
# researchOrderer("HOG", "HEALTHY", [345], 0)
# researchOrderer("HOG", "HEALTHY", [340], 0)
# researchOrderer("HOG", "HEALTHY", [335], 0)
# researchOrderer("HOG", "HEALTHY", [330], 0)
# researchOrderer("HOG", "HEALTHY", [325], 0)
# researchOrderer("HOG", "HEALTHY", [320], 0)
# researchOrderer("HOG", "HEALTHY", [315], 0)
# researchOrderer("HOG", "HEALTHY", [310], 0)
# researchOrderer("HOG", "HEALTHY", [305], 0)
# researchOrderer("HOG", "HEALTHY", [300], 0)
# researchOrderer("HOG", "HEALTHY", [295], 0)
# researchOrderer("HOG", "HEALTHY", [290], 0)
# researchOrderer("HOG", "HEALTHY", [285], 0)
# researchOrderer("HOG", "HEALTHY", [280], 0)
# researchOrderer("HOG", "HEALTHY", [275], 0)
# researchOrderer("HOG", "HEALTHY", [270], 0)
# researchOrderer("HOG", "HEALTHY", [265], 0)
# researchOrderer("HOG", "HEALTHY", [260], 0)
# researchOrderer("HOG", "HEALTHY", [255], 0)
# researchOrderer("HOG", "HEALTHY", [250], 0)
# researchOrderer("HOG", "HEALTHY", [245], 0)
# researchOrderer("HOG", "HEALTHY", [240], 0)
# researchOrderer("HOG", "HEALTHY", [235], 0)
# researchOrderer("HOG", "HEALTHY", [230], 0)
# researchOrderer("HOG", "HEALTHY", [225], 0)
# researchOrderer("HOG", "HEALTHY", [220], 0)
# researchOrderer("HOG", "HEALTHY", [215], 0)
# researchOrderer("HOG", "HEALTHY", [210], 0)
# researchOrderer("HOG", "HEALTHY", [205], 0)
# researchOrderer("HOG", "HEALTHY", [200], 0)


#############################################################
#############################################################
#############################################################


# researchOrderer("HAAR", "SICK", [1.1, 3], 0)
# researchOrderer("HAAR", "SICK", [1.2, 3], 0)
# researchOrderer("HAAR", "SICK", [1.3, 3], 0)
# researchOrderer("HAAR", "SICK", [1.4, 3], 0)
# researchOrderer("HAAR", "SICK", [1.5, 3], 0)
# researchOrderer("HAAR", "SICK", [1.6, 3], 0)
# researchOrderer("HAAR", "SICK", [1.7, 3], 0)
# researchOrderer("HAAR", "SICK", [1.8, 3], 0)
# researchOrderer("HAAR", "SICK", [1.9, 3], 0)
# researchOrderer("HAAR", "SICK", [2, 3], 0)
# researchOrderer("HAAR", "SICK", [2.1, 3], 0)
# researchOrderer("HAAR", "SICK", [2.2, 3], 0)
# researchOrderer("HAAR", "SICK", [2.3, 3], 0)
# researchOrderer("HAAR", "SICK", [2.4, 3], 0)
# researchOrderer("HAAR", "SICK", [2.5, 3], 0)
# researchOrderer("HAAR", "SICK", [2.6, 3], 0)
# researchOrderer("HAAR", "SICK", [2.7, 3], 0)
# researchOrderer("HAAR", "SICK", [2.8, 3], 0)
# researchOrderer("HAAR", "SICK", [2.9, 3], 0)
# researchOrderer("HAAR", "SICK", [3, 3], 0)
# researchOrderer("HAAR", "SICK", [3.1, 3], 0)
# researchOrderer("HAAR", "SICK", [3.2, 3], 0)
# researchOrderer("HAAR", "SICK", [3.3, 3], 0)
# researchOrderer("HAAR", "SICK", [3.4, 3], 0)
# researchOrderer("HAAR", "SICK", [3.5, 3], 0)
# researchOrderer("HAAR", "SICK", [3.6, 3], 0)
# researchOrderer("HAAR", "SICK", [3.7, 3], 0)
# researchOrderer("HAAR", "SICK", [3.8, 3], 0)
# researchOrderer("HAAR", "SICK", [3.9, 3], 0)
# researchOrderer("HAAR", "SICK", [4, 3], 0)
# researchOrderer("HAAR", "SICK", [4.1, 3], 0)
# researchOrderer("HAAR", "SICK", [4.2, 3], 0)
# researchOrderer("HAAR", "SICK", [4.3, 3], 0)
# researchOrderer("HAAR", "SICK", [4.4, 3], 0)
# researchOrderer("HAAR", "SICK", [4.5, 3], 0)
# researchOrderer("HAAR", "SICK", [4.6, 3], 0)
# researchOrderer("HAAR", "SICK", [4.7, 3], 0)
# researchOrderer("HAAR", "SICK", [4.8, 3], 0)
# researchOrderer("HAAR", "SICK", [4.9, 3], 0)
# researchOrderer("HAAR", "SICK", [5, 3], 0)
# researchOrderer("HAAR", "SICK", [5.1, 3], 0)
# researchOrderer("HAAR", "SICK", [5.2, 3], 0)
# researchOrderer("HAAR", "SICK", [5.3, 3], 0)
# researchOrderer("HAAR", "SICK", [5.4, 3], 0)
# researchOrderer("HAAR", "SICK", [5.5, 3], 0)
# researchOrderer("HAAR", "SICK", [5.6, 3], 0)
# researchOrderer("HAAR", "SICK", [5.7, 3], 0)
# researchOrderer("HAAR", "SICK", [5.8, 3], 0)
# researchOrderer("HAAR", "SICK", [5.9, 3], 0)
# researchOrderer("HAAR", "SICK", [6, 3], 0)
# researchOrderer("HAAR", "SICK", [6.1, 3], 0)
# researchOrderer("HAAR", "SICK", [6.2, 3], 0)
# researchOrderer("HAAR", "SICK", [6.3, 3], 0)
# researchOrderer("HAAR", "SICK", [6.4, 3], 0)
# researchOrderer("HAAR", "SICK", [6.5, 3], 0)
# researchOrderer("HAAR", "SICK", [6.6, 3], 0)
# researchOrderer("HAAR", "SICK", [6.7, 3], 0)
# researchOrderer("HAAR", "SICK", [6.8, 3], 0)
# researchOrderer("HAAR", "SICK", [6.9, 3], 0)
# researchOrderer("HAAR", "SICK", [7, 3], 0)
# researchOrderer("HAAR", "SICK", [7.1, 3], 0)
# researchOrderer("HAAR", "SICK", [7.2, 3], 0)
# researchOrderer("HAAR", "SICK", [7.3, 3], 0)
# researchOrderer("HAAR", "SICK", [7.4, 3], 0)
# researchOrderer("HAAR", "SICK", [7.5, 3], 0)
# researchOrderer("HAAR", "SICK", [7.6, 3], 0)
# researchOrderer("HAAR", "SICK", [7.7, 3], 0)
# researchOrderer("HAAR", "SICK", [7.8, 3], 0)
# researchOrderer("HAAR", "SICK", [7.9, 3], 0)
# researchOrderer("HAAR", "SICK", [8, 3], 0)
# researchOrderer("HAAR", "SICK", [8.1, 3], 0)
# researchOrderer("HAAR", "SICK", [8.2, 3], 0)
# researchOrderer("HAAR", "SICK", [8.3, 3], 0)
# researchOrderer("HAAR", "SICK", [8.4, 3], 0)
# researchOrderer("HAAR", "SICK", [8.5, 3], 0)
# researchOrderer("HAAR", "SICK", [8.6, 3], 0)
# researchOrderer("HAAR", "SICK", [8.7, 3], 0)
# researchOrderer("HAAR", "SICK", [8.8, 3], 0)
# researchOrderer("HAAR", "SICK", [8.9, 3], 0)
# researchOrderer("HAAR", "SICK", [9, 3], 0)
# researchOrderer("HAAR", "SICK", [9.1, 3], 0)
# researchOrderer("HAAR", "SICK", [9.2, 3], 0)
# researchOrderer("HAAR", "SICK", [9.3, 3], 0)
# researchOrderer("HAAR", "SICK", [9.4, 3], 0)
# researchOrderer("HAAR", "SICK", [9.5, 3], 0)
# researchOrderer("HAAR", "SICK", [9.6, 3], 0)
# researchOrderer("HAAR", "SICK", [9.7, 3], 0)
# researchOrderer("HAAR", "SICK", [9.8, 3], 0)
# researchOrderer("HAAR", "SICK", [9.9, 3], 0)
# researchOrderer("HAAR", "SICK", [10, 3], 0)
# researchOrderer("HAAR", "SICK", [10.1, 3], 0)
# researchOrderer("HAAR", "SICK", [10.2, 3], 0)
# researchOrderer("HAAR", "SICK", [10.3, 3], 0)
# researchOrderer("HAAR", "SICK", [10.4, 3], 0)
# researchOrderer("HAAR", "SICK", [10.5, 3], 0)
# researchOrderer("HAAR", "SICK", [10.6, 3], 0)
# researchOrderer("HAAR", "SICK", [10.7, 3], 0)
# researchOrderer("HAAR", "SICK", [10.8, 3], 0)
# researchOrderer("HAAR", "SICK", [10.9, 3], 0)
# researchOrderer("HAAR", "SICK", [11, 3], 0)
# researchOrderer("HAAR", "SICK", [11.1, 3], 0)
# 

# researchOrderer("LBP", "SICK", [1.1, 3], 0)
# researchOrderer("LBP", "SICK", [1.2, 3], 0)
# researchOrderer("LBP", "SICK", [1.3, 3], 0)
# researchOrderer("LBP", "SICK", [1.4, 3], 0)
# researchOrderer("LBP", "SICK", [1.5, 3], 0)
# researchOrderer("LBP", "SICK", [1.6, 3], 0)
# researchOrderer("LBP", "SICK", [1.7, 3], 0)
# researchOrderer("LBP", "SICK", [1.8, 3], 0)
# researchOrderer("LBP", "SICK", [1.9, 3], 0)
# researchOrderer("LBP", "SICK", [2, 3], 0)
# researchOrderer("LBP", "SICK", [2.1, 3], 0)
# researchOrderer("LBP", "SICK", [2.2, 3], 0)
# researchOrderer("LBP", "SICK", [2.3, 3], 0)
# researchOrderer("LBP", "SICK", [2.4, 3], 0)
# researchOrderer("LBP", "SICK", [2.5, 3], 0)
# researchOrderer("LBP", "SICK", [2.6, 3], 0)
# researchOrderer("LBP", "SICK", [2.7, 3], 0)
# researchOrderer("LBP", "SICK", [2.8, 3], 0)
# researchOrderer("LBP", "SICK", [2.9, 3], 0)
# researchOrderer("LBP", "SICK", [3, 3], 0)
# researchOrderer("LBP", "SICK", [3.1, 3], 0)
# researchOrderer("LBP", "SICK", [3.2, 3], 0)
# researchOrderer("LBP", "SICK", [3.3, 3], 0)
# researchOrderer("LBP", "SICK", [3.4, 3], 0)
# researchOrderer("LBP", "SICK", [3.5, 3], 0)
# researchOrderer("LBP", "SICK", [3.6, 3], 0)
# researchOrderer("LBP", "SICK", [3.7, 3], 0)
# researchOrderer("LBP", "SICK", [3.8, 3], 0)
# researchOrderer("LBP", "SICK", [3.9, 3], 0)
# researchOrderer("LBP", "SICK", [4, 3], 0)
# researchOrderer("LBP", "SICK", [4.1, 3], 0)
# researchOrderer("LBP", "SICK", [4.2, 3], 0)
# researchOrderer("LBP", "SICK", [4.3, 3], 0)
# researchOrderer("LBP", "SICK", [4.4, 3], 0)
# researchOrderer("LBP", "SICK", [4.5, 3], 0)
# researchOrderer("LBP", "SICK", [4.6, 3], 0)
# researchOrderer("LBP", "SICK", [4.7, 3], 0)
# researchOrderer("LBP", "SICK", [4.8, 3], 0)
# researchOrderer("LBP", "SICK", [4.9, 3], 0)
# researchOrderer("LBP", "SICK", [5, 3], 0)
# researchOrderer("LBP", "SICK", [5.1, 3], 0)
# researchOrderer("LBP", "SICK", [5.2, 3], 0)
# researchOrderer("LBP", "SICK", [5.3, 3], 0)
# researchOrderer("LBP", "SICK", [5.4, 3], 0)
# researchOrderer("LBP", "SICK", [5.5, 3], 0)
# researchOrderer("LBP", "SICK", [5.6, 3], 0)
# researchOrderer("LBP", "SICK", [5.7, 3], 0)
# researchOrderer("LBP", "SICK", [5.8, 3], 0)
# researchOrderer("LBP", "SICK", [5.9, 3], 0)
# researchOrderer("LBP", "SICK", [6, 3], 0)
# researchOrderer("LBP", "SICK", [6.1, 3], 0)
# researchOrderer("LBP", "SICK", [6.2, 3], 0)
# researchOrderer("LBP", "SICK", [6.3, 3], 0)
# researchOrderer("LBP", "SICK", [6.4, 3], 0)
# researchOrderer("LBP", "SICK", [6.5, 3], 0)
# researchOrderer("LBP", "SICK", [6.6, 3], 0)
# researchOrderer("LBP", "SICK", [6.7, 3], 0)
# researchOrderer("LBP", "SICK", [6.8, 3], 0)
# researchOrderer("LBP", "SICK", [6.9, 3], 0)
# researchOrderer("LBP", "SICK", [7, 3], 0)
# researchOrderer("LBP", "SICK", [7.1, 3], 0)
# researchOrderer("LBP", "SICK", [7.2, 3], 0)
# researchOrderer("LBP", "SICK", [7.3, 3], 0)
# researchOrderer("LBP", "SICK", [7.4, 3], 0)
# researchOrderer("LBP", "SICK", [7.5, 3], 0)
# researchOrderer("LBP", "SICK", [7.6, 3], 0)
# researchOrderer("LBP", "SICK", [7.7, 3], 0)
# researchOrderer("LBP", "SICK", [7.8, 3], 0)
# researchOrderer("LBP", "SICK", [7.9, 3], 0)
# researchOrderer("LBP", "SICK", [8, 3], 0)
# researchOrderer("LBP", "SICK", [8.1, 3], 0)
# researchOrderer("LBP", "SICK", [8.2, 3], 0)
# researchOrderer("LBP", "SICK", [8.3, 3], 0)
# researchOrderer("LBP", "SICK", [8.4, 3], 0)
# researchOrderer("LBP", "SICK", [8.5, 3], 0)
# researchOrderer("LBP", "SICK", [8.6, 3], 0)
# researchOrderer("LBP", "SICK", [8.7, 3], 0)
# researchOrderer("LBP", "SICK", [8.8, 3], 0)
# researchOrderer("LBP", "SICK", [8.9, 3], 0)
# researchOrderer("LBP", "SICK", [9, 3], 0)
# researchOrderer("LBP", "SICK", [9.1, 3], 0)
# researchOrderer("LBP", "SICK", [9.2, 3], 0)
# researchOrderer("LBP", "SICK", [9.3, 3], 0)
# researchOrderer("LBP", "SICK", [9.4, 3], 0)
# researchOrderer("LBP", "SICK", [9.5, 3], 0)
# researchOrderer("LBP", "SICK", [9.6, 3], 0)
# researchOrderer("LBP", "SICK", [9.7, 3], 0)
# researchOrderer("LBP", "SICK", [9.8, 3], 0)
# researchOrderer("LBP", "SICK", [9.9, 3], 0)
# researchOrderer("LBP", "SICK", [10, 3], 0)
# researchOrderer("LBP", "SICK", [10.1, 3], 0)
# researchOrderer("LBP", "SICK", [10.2, 3], 0)
# researchOrderer("LBP", "SICK", [10.3, 3], 0)
# researchOrderer("LBP", "SICK", [10.4, 3], 0)
# researchOrderer("LBP", "SICK", [10.5, 3], 0)
# researchOrderer("LBP", "SICK", [10.6, 3], 0)
# researchOrderer("LBP", "SICK", [10.7, 3], 0)
# researchOrderer("LBP", "SICK", [10.8, 3], 0)
# researchOrderer("LBP", "SICK", [10.9, 3], 0)
# researchOrderer("LBP", "SICK", [11, 3], 0)
# researchOrderer("LBP", "SICK", [11.1, 3], 0)
# 


# researchOrderer("HOG", "SICK", [700], 0)
# researchOrderer("HOG", "SICK", [695], 0)
# researchOrderer("HOG", "SICK", [690], 0)
# researchOrderer("HOG", "SICK", [685], 0)
# researchOrderer("HOG", "SICK", [680], 0)
# researchOrderer("HOG", "SICK", [675], 0)
# researchOrderer("HOG", "SICK", [670], 0)
# researchOrderer("HOG", "SICK", [665], 0)
# researchOrderer("HOG", "SICK", [660], 0)
# researchOrderer("HOG", "SICK", [655], 0)
# researchOrderer("HOG", "SICK", [650], 0)
# researchOrderer("HOG", "SICK", [645], 0)
# researchOrderer("HOG", "SICK", [640], 0)
# researchOrderer("HOG", "SICK", [635], 0)
# researchOrderer("HOG", "SICK", [630], 0)
# researchOrderer("HOG", "SICK", [625], 0)
# researchOrderer("HOG", "SICK", [620], 0)
# researchOrderer("HOG", "SICK", [615], 0)
# researchOrderer("HOG", "SICK", [610], 0)
# researchOrderer("HOG", "SICK", [605], 0)
# researchOrderer("HOG", "SICK", [600], 0)
# researchOrderer("HOG", "SICK", [595], 0)
# researchOrderer("HOG", "SICK", [590], 0)
# researchOrderer("HOG", "SICK", [585], 0)
# researchOrderer("HOG", "SICK", [580], 0)
# researchOrderer("HOG", "SICK", [575], 0)
# researchOrderer("HOG", "SICK", [570], 0)
# researchOrderer("HOG", "SICK", [565], 0)
# researchOrderer("HOG", "SICK", [560], 0)
# researchOrderer("HOG", "SICK", [555], 0)
# researchOrderer("HOG", "SICK", [550], 0)
# researchOrderer("HOG", "SICK", [545], 0)
# researchOrderer("HOG", "SICK", [540], 0)
# researchOrderer("HOG", "SICK", [535], 0)
# researchOrderer("HOG", "SICK", [530], 0)
# researchOrderer("HOG", "SICK", [525], 0)
# researchOrderer("HOG", "SICK", [520], 0)
# researchOrderer("HOG", "SICK", [515], 0)
# researchOrderer("HOG", "SICK", [510], 0)
# researchOrderer("HOG", "SICK", [505], 0)
# researchOrderer("HOG", "SICK", [500], 0)
# researchOrderer("HOG", "SICK", [495], 0)
# researchOrderer("HOG", "SICK", [490], 0)
# researchOrderer("HOG", "SICK", [485], 0)
# researchOrderer("HOG", "SICK", [480], 0)
# researchOrderer("HOG", "SICK", [475], 0)
# researchOrderer("HOG", "SICK", [470], 0)
# researchOrderer("HOG", "SICK", [465], 0)
# researchOrderer("HOG", "SICK", [460], 0)
# researchOrderer("HOG", "SICK", [455], 0)
# researchOrderer("HOG", "SICK", [450], 0)
# researchOrderer("HOG", "SICK", [445], 0)
# researchOrderer("HOG", "SICK", [440], 0)
# researchOrderer("HOG", "SICK", [435], 0)
# researchOrderer("HOG", "SICK", [430], 0)
# researchOrderer("HOG", "SICK", [425], 0)
# researchOrderer("HOG", "SICK", [420], 0)
# researchOrderer("HOG", "SICK", [415], 0)
# researchOrderer("HOG", "SICK", [410], 0)
# researchOrderer("HOG", "SICK", [405], 0)
# researchOrderer("HOG", "SICK", [400], 0)
################################################################################

# researchOrderer("HOG", "SICK", [395], 0)
# researchOrderer("HOG", "SICK", [390], 0)
# researchOrderer("HOG", "SICK", [385], 0)
# researchOrderer("HOG", "SICK", [380], 0)
# researchOrderer("HOG", "SICK", [375], 0)
# researchOrderer("HOG", "SICK", [370], 0)
# researchOrderer("HOG", "SICK", [365], 0)
# researchOrderer("HOG", "SICK", [360], 0)
# researchOrderer("HOG", "SICK", [355], 0)
# researchOrderer("HOG", "SICK", [350], 0)
# researchOrderer("HOG", "SICK", [345], 0)
# researchOrderer("HOG", "SICK", [340], 0)
# researchOrderer("HOG", "SICK", [335], 0)
# researchOrderer("HOG", "SICK", [330], 0)
# researchOrderer("HOG", "SICK", [325], 0)
# researchOrderer("HOG", "SICK", [320], 0)
# researchOrderer("HOG", "SICK", [315], 0)
# researchOrderer("HOG", "SICK", [310], 0)
# researchOrderer("HOG", "SICK", [305], 0)
# researchOrderer("HOG", "SICK", [300], 0)

# researchOrderer("HOG", "SICK", [295], 0)
# researchOrderer("HOG", "SICK", [290], 0)
# researchOrderer("HOG", "SICK", [285], 0)
# researchOrderer("HOG", "SICK", [280], 0)
# researchOrderer("HOG", "SICK", [275], 0)
# researchOrderer("HOG", "SICK", [270], 0)
# researchOrderer("HOG", "SICK", [265], 0)
# researchOrderer("HOG", "SICK", [260], 0)
# researchOrderer("HOG", "SICK", [255], 0)
# researchOrderer("HOG", "SICK", [250], 0)
# researchOrderer("HOG", "SICK", [245], 0)
# researchOrderer("HOG", "SICK", [240], 0)
# researchOrderer("HOG", "SICK", [235], 0)
# researchOrderer("HOG", "SICK", [230], 0)
# researchOrderer("HOG", "SICK", [225], 0)
# researchOrderer("HOG", "SICK", [220], 0)
# researchOrderer("HOG", "SICK", [215], 0)
# researchOrderer("HOG", "SICK", [210], 0)
# researchOrderer("HOG", "SICK", [205], 0)
# researchOrderer("HOG", "SICK", [200], 0)


#
# researchOrderer("CNNDLIB", "SICK", [700, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [695, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [690, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [685, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [680, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [675, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [670, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [665, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [660, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [655, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [650, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [645, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [640, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [635, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [630, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [625, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [620, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [615, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [610, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [605, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [600, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [595, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [590, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [585, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [580, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [575, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [570, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [565, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [560, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [555, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [550, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [545, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [540, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [535, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [530, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [525, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [520, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [515, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [510, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [505, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [500, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [495, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [490, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [485, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [480, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [475, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [470, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [465, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [460, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [455, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [450, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [445, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [440, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [435, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [430, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [425, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [420, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [415, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [410, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [405, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [400, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [395, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [390, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [385, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [380, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [375, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [370, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [365, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [360, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [355, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [350, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [345, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [340, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [335, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [330, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [325, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [320, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [315, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [310, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [305, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [300, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [295, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [290, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [285, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [280, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [275, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [270, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [265, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [260, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [255, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [250, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [245, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [240, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [235, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [230, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [225, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [220, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [215, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [210, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [205, 1], 0)
# researchOrderer("CNNDLIB", "SICK", [200, 1], 0)


researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.01], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.02], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.03], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.04], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.05], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.06], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.07], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.08], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.09], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.1], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.11], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.12], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.13], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.14], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.15], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.16], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.17], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.18], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.19], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.2], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.21], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.22], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.23], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.24], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.25], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.26], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.27], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.28], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.29], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.3], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.31], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.32], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.33], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.34], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.35], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.36], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.37], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.38], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.39], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.4], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.41], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.42], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.43], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.44], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.45], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.46], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.47], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.48], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.49], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.5], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.51], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.52], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.53], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.54], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.55], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.56], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.57], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.58], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.59], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.6], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.61], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.62], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.63], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.64], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.65], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.66], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.67], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.68], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.69], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.7], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.71], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.72], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.73], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.74], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.75], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.76], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.77], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.78], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.79], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.8], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.81], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.82], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.83], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.84], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.85], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.86], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.87], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.88], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.89], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.9], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.91], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.92], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.93], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.94], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.95], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.96], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.97], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.98], 0)
researchOrderer("MTCNN", "SICK", [[0.2, 0.5, 0.8], 0.99], 0)
file.close()
