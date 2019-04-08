#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:00:16 2017

@author: tomislav
"""

import cv2
import numpy as np
from obj_detector.detector import (obj_detector, bow_features,
                                   get_extract_detect, get_flann_matcher,
                                   get_bow_extractor)
from obj_detector.pyramid import pyramid
from obj_detector.non_maximum import non_maximum_suppression_fast as nms
from obj_detector.sliding_window import sliding_window
from obj_detector.contour_detection import drawContourWindow
import time


def in_range(number, test, thresh=0.2):
    return abs(number - test) < thresh


time_start = time.time()
test_image = 'testImages/test2/testImage-0048.bmp'

# Build svm and bow extractor from image samples
# svm, extractor_bow = obj_detector()


# Load svm from file
svm = cv2.ml.SVM_load('svm_data_ind.xml')
# Load BOW vocabulary from file
fs = cv2.FileStorage('bow_vocabulary_ind.xml', cv2.FileStorage_READ)
node = fs.getNode('bow-vocabulary-ind')
voc = node.mat()
fs.release()
# FLANN matcher
matcher = get_flann_matcher()
# Feature extractor adn detector (SIFT)
extract, detect = get_extract_detect()
# BOW extractor
extractor_bow = get_bow_extractor(extract, matcher)
extractor_bow.setVocabulary(voc)


w, h = 160, 160
img = cv2.imread(test_image)
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rectangles1 = []
rectangles2 = []
counter = 1
scaleFactor = 1.1
scale = 1
font = cv2.FONT_HERSHEY_PLAIN
score = 0

for resized in pyramid(imgGray, scaleFactor, minSize=(200, 200)):
    scale = float(imgGray.shape[1]) / float(resized.shape[1])
    for (x, y, roi) in sliding_window(resized, stepSize=8, windowSize=(w, h)):
        if roi.shape[1] != w or roi.shape[0] != h:
            continue
        try:
            bf = bow_features(roi, extractor_bow, detect)
            _, result = svm.predict(bf)
            a, res = svm.predict(bf, flags=cv2.ml.STAT_MODEL_RAW_OUTPUT)
            print 'Class: %d, Score: %f' % (result[0][0], res[0][0])
            score = res[0][0]
            if result[0][0] == 2:
                if score < -0.2:
                    (rx, ry, rx2, ry2) = (
                            int(x * scale), int(y * scale), int((x+w) * scale),
                            int((y+h) * scale))
                    rectangles2.append([rx, ry, rx2, ry2, abs(score)])
            if result[0][0] == 1:
                if score > 0.721:
                    (rx, ry, rx2, ry2) = (
                            int(x * scale), int(y * scale), int((x+w) * scale),
                            int((y+h) * scale))
                    rectangles1.append([rx, ry, rx2, ry2, abs(score)])
        except:
            pass

        counter += 1
        # Show sliding windows
#        clone = resized.copy()
#        cv2.rectangle(clone, (int(x), int(y)), (int(x + w), int(y + h)),
#                      (255, 0, 0), 2)
#        cv2.putText(clone, '%f' % score, (int(x), int(y)), font, 1,
#                    (0, 255, 0), 2)
#        cv2.imshow('Window', clone)
#        cv2.waitKey(1)

windows1 = np.array(rectangles1)
windows2 = np.array(rectangles2)
boxes1 = nms(windows1, 0.15)
boxes2 = nms(windows2, 0.15)

for (x, y, x2, y2, score) in boxes1:
    print x, y, x2, y2, score
    cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0, 128, 255), 2)
    cv2.putText(img, '%f' % score, (int(x), int(y)), font, 1, (0, 128, 255), 2)

for (x, y, x2, y2, score) in boxes2:
    print x, y, x2, y2, score
    cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (153, 51, 255), 2)
    cv2.putText(img, '%f' % score, (int(x), int(y)), font, 1, (153, 51, 255), 2)

cv2.imshow('img', img)
time_elapsed = time.time() - time_start
print "Vrijeme izvodenja: {} sec".format(time_elapsed)
drwCnt = drawContourWindow(imgGray, img)

for (x, y, x2, y2, score) in boxes1:
    drwCnt(x, y, x2, y2, 'cep')

for (x, y, x2, y2, score) in boxes2:
    drwCnt(x, y, x2, y2, 'kolj')

cv2.destroyWindow('contour')
cv2.imshow('img', img)
cv2.waitKey()
cv2.destroyAllWindows()
