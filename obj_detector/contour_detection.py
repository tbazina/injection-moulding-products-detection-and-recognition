#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 18:48:11 2017

@author: tomislav
"""

import numpy as np
import cv2


def ContourPlotter(imgGray, imgColor, bilFilterSize=6,
                   cannyThresh1=27, cannyThresh2=30, closeKernelSize=11,
                   areaThresh=1000):
    # Defining ROI
    gray = imgGray
    color = imgColor
    # Biltateral blurring
    gray = cv2.bilateralFilter(gray, d=bilFilterSize, sigmaColor=100,
                               sigmaSpace=100)
    # Finding edges with Canny
    edges = cv2.Canny(gray, cannyThresh1, cannyThresh2,
                      apertureSize=3, L2gradient=False)
    # Closing contour with morphological operation Closing
    morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closeKernelSize,
                                                             closeKernelSize))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morphKernel)
    _, conts, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    # Discard all bad contours
    gen = [cnt for cnt in conts if cv2.contourArea(cnt) > areaThresh]
    cv2.drawContours(color, gen, -1, (255, 0, 0), 1)
    return [gen, color]


class drawContourWindow(object):
    def __init__(self, imgGray, imgColor):
        self.imgGray = imgGray
        self.imgColor = imgColor
        self.bilFilterSize = 5
        self.cannyThresh1 = 27
        self.cannyThresh2 = 60
        self.closeKernelSize = 5
        self.areaThresh = 1000

        cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('FilterSize', 'contour', 5, 50, self.update)
        cv2.createTrackbar('Thresh1', 'contour', 27, 200, self.update)
        cv2.createTrackbar('Thresh2', 'contour', 60, 200, self.update)
        cv2.createTrackbar('KernelSize', 'contour', 5, 50, self.update)
        cv2.createTrackbar('areaThresh', 'contour', 1000, 50000, self.update)

    def __call__(self, x, y, x2, y2, part='kolj'):
        self.x = int(max(x - 0.55*x, 0))
        self.y = int(max(y - 0.55*y, 0))
        self.x2 = int(x2 + 0.2*x2)
        self.y2 = int(y2 + 0.2*y2)
        self.cnt, self.img = ContourPlotter(
                self.imgGray.copy()[self.y:self.y2, self.x:self.x2],
                self.imgColor.copy()[self.y:self.y2, self.x:self.x2],
                self.bilFilterSize, self.cannyThresh1,
                self.cannyThresh2, self.closeKernelSize, self.areaThresh)
        cv2.imshow('contour', self.img)
        cv2.waitKey()

        if len(self.cnt) != 0:
            # Draw contour
            if part == 'kolj':
                cv2.drawContours(self.imgColor[self.y:self.y2,
                                               self.x:self.x2],
                                 self.cnt, -1, (153, 51, 255), 2)
            elif part == 'cep':
                cv2.drawContours(self.imgColor[self.y:self.y2,
                                               self.x:self.x2],
                                 self.cnt, -1, (0, 128, 255), 2)
            else:
                print 'Unesi cep ili kolj!'
            for cnt in self.cnt:
                # Moments
                M = cv2.moments(cnt)
                # Centroid
                cx = np.int(np.divide(np.int(M['m10']), np.int(M['m00'])))
                cy = np.int(np.divide(np.int(M['m01']), np.int(M['m00'])))
                cv2.circle(self.imgColor[self.y:self.y2, self.x:self.x2],
                           (cx, cy), 2, (153, 0, 153), 2)
                if part == 'kolj':
                    # Orientation
                    angle = OrientationAngleKolj(M['mu11'], M['mu20'],
                                                 M['mu02'], M['mu03'],
                                                 M['mu30'])
                elif part == 'cep':
                    angle = OrientationAngleCep(M['mu11'], M['mu20'],
                                                M['mu02'])
                # Axes
                angleMajor = angle
                lengthMajor = 25.
                majorX = np.int(cx + lengthMajor * np.cos(angleMajor))
                majorY = np.int(cy + lengthMajor * np.sin(angleMajor))
                cv2.line(self.imgColor[self.y:self.y2, self.x:self.x2],
                         (cx, cy), (majorX, majorY), (0, 0, 240), 2)
                lengthMinor = 15.
                minorX = np.int(cx + lengthMinor * np.sin(angleMajor))
                minorY = np.int(cy - lengthMinor * np.cos(angleMajor))
                cv2.line(self.imgColor[self.y:self.y2, self.x:self.x2],
                         (cx, cy), (minorX, minorY), (220, 0, 0), 2)

    def update(self, pos):
        bilFilterSize = cv2.getTrackbarPos('FilterSize', 'contour')
        if bilFilterSize > 0:
            self.bilFilterSize = bilFilterSize
        self.cannyThresh1 = cv2.getTrackbarPos('Thresh1', 'contour')
        self.cannyThresh2 = cv2.getTrackbarPos('Thresh2', 'contour')
        self.areaThresh = cv2.getTrackbarPos('areaThresh', 'contour')
        closeKernelSize = cv2.getTrackbarPos('KernelSize', 'contour')
        if closeKernelSize % 2 != 0:
            self.closeKernelSize = closeKernelSize

        self.cnt, self.img = ContourPlotter(
                self.imgGray.copy()[self.y:self.y2, self.x:self.x2],
                self.imgColor.copy()[self.y:self.y2, self.x:self.x2],
                self.bilFilterSize, self.cannyThresh1,
                self.cannyThresh2, self.closeKernelSize, self.areaThresh)
        cv2.imshow('contour', self.img)


def OrientationAngleKolj(mu11, mu20, mu02, mu03, mu30):
    angle = 0.5 * np.arctan(np.divide(2. * mu11, mu20 - mu02))
    if mu20 > mu02 and mu30 > 0:
        angle += 0.97*(np.pi / 4.) - np.pi/2.
    elif mu20 > mu02 and mu30 < 0:
        angle += 0.97*(np.pi / 4.) - np.pi/2. + np.pi
    elif mu20 < mu02 and mu30 < 0:
        angle += 0.97*(np.pi / 4.) + np.pi
    elif mu20 < mu02 and mu30 > 0:
        angle += 0.97*(np.pi / 4.)
    return angle


def OrientationAngleCep(mu11, mu20, mu02):
    angle = 0.5 * np.arctan(np.divide(2. * mu11, mu20 - mu02))
    return angle


#img = cv2.imread('../trainImages/kolj/kolj-0016.pgm', 0)
#imgColor = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

"""
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
imgColorCnt = imgColor.copy()

img = cv2.bilateralFilter(img, d=6, sigmaColor=100, sigmaSpace=100)
edges = cv2.Canny(img, 27, 30, apertureSize=3, L2gradient=False)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
im2, conts, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(imgColor, conts, -1, (255, 0, 0), 1)
gen = [cnt for cnt in conts if cv2.contourArea(cnt) > 1000]
cv2.drawContours(imgColor, gen, -1, (255, 0, 0), 1)
for cnt in gen:
    # Moments
    M = cv2.moments(cnt)
    # Centroid
    cx = np.int(np.divide(np.int(M['m10']), np.int(M['m00'])))
    cy = np.int(np.divide(np.int(M['m01']), np.int(M['m00'])))
    cv2.circle(imgColorCnt, (cx, cy), 3, (255, 255, 0), 1)

    # Orientation
    angle = OrientationAngle(M['mu11'], M['mu20'], M['mu02'], M['mu03'],
                             M['mu30'])
    print 'angle = {}'.format(angle)

    # Axes
    angleMajor = angle
    lengthMajor = 25.
    majorX = np.int(cx + lengthMajor * np.cos(angleMajor))
    majorY = np.int(cy + lengthMajor * np.sin(angleMajor))
    cv2.line(imgColorCnt, (cx, cy), (majorX, majorY), (0, 0, 255), 2)
    lengthMinor = 15.
    angleMinor = angleMajor - np.pi / 2.
    minorX = np.int(cx + lengthMinor * np.sin(angleMajor))
    minorY = np.int(cy - lengthMinor * np.cos(angleMajor))
    cv2.line(imgColorCnt, (cx, cy), (minorX, minorY), (255, 0, 0), 2)

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(imgColor, (x, y), (x+w, y+h), (0, 255, 0), 2)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(imgColor, [box], 0, (0, 0, 255), 1)

    hull = cv2.convexHull(cnt)
    cv2.drawContours(imgColorCnt, [hull], 0, (0, 255, 0), 1)
#
#    epsilon = 0.01 * cv2.arcLength(cnt, True)
#    approx = cv2.approxPolyDP(cnt, epsilon, True)
#    cv2.drawContours(imgColor, [approx], 0, (255, 255, 0), 1)

cv2.imshow('image', img)
cv2.imshow('color', imgColor)
cv2.imshow('foreground', imgColorCnt)
cv2.imshow('Canny', edges)
"""

#drwCnt = drawContourWindow(img, imgColor)
#drwCnt(0, 0, img.shape[1], img.shape[1], 'cep')
#cv2.imshow('img', imgColor)

#cv2.waitKey()
#cv2.destroyAllWindows()

