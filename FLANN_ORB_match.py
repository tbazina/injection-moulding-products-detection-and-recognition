#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:40:51 2017

@author: tomislav
"""

import cv2
import numpy as np


class ORBWindow(object):
    def __init__(self, queryImage, trainingImage):
        self.queryImage = queryImage
        self.trainingImage = trainingImage
        self.orbParameters = {
                'nfeatures': 1000,
                'scaleFactor': 1.11,
                'nlevels': 30,
                'edgeThreshold': 8,  # roughly match the patchSize parameter
                'firstLevel': 0,  # 0
                'WTA_K': 3,  # 2, 3, 4
                'scoreType': cv2.ORB_HARRIS_SCORE,  # cv2.ORB_FAST_SCORE
                'patchSize': 73
                }
        self.orb = cv2.ORB_create(**self.orbParameters)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.queryImage, None)
        self.kp2, self.des2 = self.orb.detectAndCompute(self.trainingImage,
                                                        None)
        self.des1, self.des2 = np.float32(self.des1), np.float32(self.des2)

        self.FLANN_INDEX_LSH = 0
        self.indexParams = dict(algorithm=self.FLANN_INDEX_LSH, table_number=6,
                                key_size=12, multi_probe_level=1)
        self.searchParams = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.indexParams, self.searchParams)

        self.matches = self.flann.knnMatch(self.des1, self.des2, k=2)
        self.matchesMaks = [[0, 0] for i in xrange(len(self.matches))]
        for i, (m, n) in enumerate(self.matches):
            if m.distance < 0.7 * n.distance:
                self.matchesMaks[i] = [1, 0]
        self.drawParams = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=self.matchesMaks,
                               flags=0)

        self.resultImage = cv2.drawMatchesKnn(self.queryImage, self.kp1,
                                              self.trainingImage, self.kp2,
                                              self.matches, None,
                                              **self.drawParams)
        cv2.imshow('result', self.resultImage)
        cv2.createTrackbar('nfeatures', 'result', 1000, 5000, self.update)
        cv2.createTrackbar('scaleFactor', 'result', 11, 80, self.update)
        cv2.createTrackbar('nlevels', 'result', 30, 50, self.update)
        cv2.createTrackbar('edgeThreshold', 'result', 8, 100, self.update)
        cv2.createTrackbar('WTA_K', 'result', 3, 4, self.update)
        cv2.createTrackbar('patchSize', 'result', 73, 100, self.update)

    def update(self, pos):
        nfeatures = cv2.getTrackbarPos('nfeatures', 'result')
        if nfeatures > 5:
            self.nfeatures = nfeatures
        scaleFactor = cv2.getTrackbarPos('scaleFactor', 'result')
        if scaleFactor >= 2:
            self.scaleFactor = 1. + scaleFactor / 100.
        nlevels = cv2.getTrackbarPos('nlevels', 'result')
        if nlevels > 0:
            self.nlevels = nlevels
        edgeThreshold = cv2.getTrackbarPos('edgeThreshold', 'result')
        if edgeThreshold > 0:
            self.edgeThreshold = edgeThreshold
        WTA_K = cv2.getTrackbarPos('WTA_K', 'result')
        if WTA_K >= 2:
            self.WTA_K = WTA_K
        patchSize = cv2.getTrackbarPos('patchSize', 'result')
        if patchSize > 0:
            self.patchSize = patchSize

        orbParameters = {
                'nfeatures': self.nfeatures,
                'scaleFactor': self.scaleFactor,
                'nlevels': self.nlevels,
                'edgeThreshold': self.edgeThreshold,
                # roughly match the patchSize parameter
                'firstLevel': 0,  # 0
                'WTA_K': self.WTA_K,  # 2, 3, 4
                'scoreType': cv2.ORB_HARRIS_SCORE,  # cv2.ORB_FAST_SCORE
                'patchSize': self.patchSize
                }

        orb = cv2.ORB_create(**orbParameters)
        kp1, des1 = orb.detectAndCompute(self.queryImage, None)
        kp2, des2 = orb.detectAndCompute(self.trainingImage,
                                         None)
        des1, des2 = np.float32(des1), np.float32(des2)

        flann = cv2.FlannBasedMatcher(self.indexParams, self.searchParams)

        matches = flann.knnMatch(des1, des2, k=2)
        matchesMaks = [[0, 0] for i in xrange(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMaks[i] = [1, 0]
        drawParams = dict(matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          matchesMask=matchesMaks,
                          flags=0)

        resultImage = cv2.drawMatchesKnn(self.queryImage, kp1,
                                         self.trainingImage, kp2,
                                         matches, None,
                                         **drawParams)
        cv2.imshow('result', resultImage)


class SIFTWindow(object):
    def __init__(self, queryImage, trainingImage):
        self.queryImage = queryImage
        self.trainingImage = trainingImage
        self.siftParameters = {
                'nfeatures': 1000,
                'nOctaveLayers': 7,
                'contrastThreshold': 0.018,  # larger threshold, less features
                'edgeThreshold': 40,  # larger threshold, more features
                'sigma': 1.56  # weak camera, reduce the number
                }
        self.sift = cv2.xfeatures2d.SIFT_create(**self.siftParameters)
        self.kp1, self.des1 = self.sift.detectAndCompute(self.queryImage, None)
        self.kp2, self.des2 = self.sift.detectAndCompute(self.trainingImage,
                                                         None)
        self.des1, self.des2 = np.float32(self.des1), np.float32(self.des2)

        self.FLANN_INDEX_KDTREE = 0
        self.indexParams = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.searchParams = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.indexParams, self.searchParams)

        self.matches = self.flann.knnMatch(self.des1, self.des2, k=2)
        self.matchesMaks = [[0, 0] for i in xrange(len(self.matches))]
        for i, (m, n) in enumerate(self.matches):
            if m.distance < 0.7 * n.distance:
                self.matchesMaks[i] = [1, 0]
        self.drawParams = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=self.matchesMaks,
                               flags=0)

        self.resultImage = cv2.drawMatchesKnn(self.queryImage, self.kp1,
                                              self.trainingImage, self.kp2,
                                              self.matches, None,
                                              **self.drawParams)
        cv2.imshow('result', self.resultImage)
        cv2.createTrackbar('nfeatures', 'result', 1000, 1000, self.update)
        cv2.createTrackbar('nOctaveLayers', 'result', 7, 20, self.update)
        cv2.createTrackbar('contrastThreshold', 'result', 18, 100,
                           self.update)
        cv2.createTrackbar('edgeThreshold', 'result', 40, 60, self.update)
        cv2.createTrackbar('sigma', 'result', 56, 100, self.update)

    def update(self, pos):
        nfeatures = cv2.getTrackbarPos('nfeatures', 'result')
        if nfeatures > 5:
            self.nfeatures = nfeatures
        nOctaveLayers = cv2.getTrackbarPos('nOctaveLayers', 'result')
        if nOctaveLayers >= 1:
            self.nOctaveLayers = nOctaveLayers
        contrastThreshold = cv2.getTrackbarPos('contrastThreshold', 'result')
        if contrastThreshold > 0:
            self.contrastThreshold = contrastThreshold / 1000.
        edgeThreshold = cv2.getTrackbarPos('edgeThreshold', 'result')
        if edgeThreshold > 0:
            self.edgeThreshold = edgeThreshold
        sigma = cv2.getTrackbarPos('sigma', 'result')
        if sigma > 1:
            self.sigma = 1 + sigma / 100.

        siftParameters = {
                'nfeatures': self.nfeatures,
                'nOctaveLayers': self.nOctaveLayers,
                'contrastThreshold': self.contrastThreshold,
                'edgeThreshold': self.edgeThreshold,
                'sigma': self.sigma,
                }

        sift = cv2.xfeatures2d.SIFT_create(**siftParameters)
        kp1, des1 = sift.detectAndCompute(self.queryImage, None)
        kp2, des2 = sift.detectAndCompute(self.trainingImage, None)
        des1, des2 = np.float32(des1), np.float32(des2)

        flann = cv2.FlannBasedMatcher(self.indexParams, self.searchParams)

        matches = flann.knnMatch(des1, des2, k=2)
        matchesMaks = [[0, 0] for i in xrange(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMaks[i] = [1, 0]
        drawParams = dict(matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          matchesMask=matchesMaks,
                          flags=0)

        resultImage = cv2.drawMatchesKnn(self.queryImage, kp1,
                                         self.trainingImage, kp2,
                                         matches, None,
                                         **drawParams)
        cv2.imshow('result', resultImage)


class SURFWindow(object):
    def __init__(self, queryImage, trainingImage):
        self.queryImage = queryImage
        self.trainingImage = trainingImage
        self.surfParameters = {
                'hessianThreshold': 550,  # larger value, less features
                # 300-500 good value # 870
                'nOctaves': 2,  # number of pyramid octaves
                'nOctaveLayers': 20,  # number of layers within octave
                'extended': False,  # 128/64 elements descriptors
                'upright': False  # compute orientation or not
                }
        self.surf = cv2.xfeatures2d.SURF_create(**self.surfParameters)
        self.kp1, self.des1 = self.surf.detectAndCompute(self.queryImage, None)
        self.kp2, self.des2 = self.surf.detectAndCompute(self.trainingImage,
                                                         None)
        self.des1, self.des2 = np.float32(self.des1), np.float32(self.des2)

        self.FLANN_INDEX_KDTREE = 0
        self.indexParams = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=5)
        self.searchParams = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(self.indexParams, self.searchParams)

        self.matches = self.flann.knnMatch(self.des1, self.des2, k=2)
        self.matchesMaks = [[0, 0] for i in xrange(len(self.matches))]
        for i, (m, n) in enumerate(self.matches):
            if m.distance < 0.7 * n.distance:
                self.matchesMaks[i] = [1, 0]
        self.drawParams = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=self.matchesMaks,
                               flags=0)

        self.resultImage = cv2.drawMatchesKnn(self.queryImage, self.kp1,
                                              self.trainingImage, self.kp2,
                                              self.matches, None,
                                              **self.drawParams)
        cv2.imshow('result', self.resultImage)
        cv2.createTrackbar('hessianThreshold', 'result', 550, 2000,
                           self.update)
        cv2.createTrackbar('nOctaves', 'result', 2, 25, self.update)
        cv2.createTrackbar('nOctaveLayers', 'result', 20, 40, self.update)
        cv2.createTrackbar('extended', 'result', 0, 1, self.update)
        cv2.createTrackbar('trees', 'result', 5, 20, self.update)
        cv2.createTrackbar('checks', 'result', 50, 200, self.update)
#        cv2.imwrite('SURF_rotation_kolj.jpg', self.resultImage)

    def update(self, pos):
        self.hessianThreshold = cv2.getTrackbarPos('hessianThreshold',
                                                   'result')
        self.nOctaves = cv2.getTrackbarPos('nOctaves', 'result')
        self.nOctaveLayers = cv2.getTrackbarPos('nOctaveLayers', 'result')
        self.extended = bool(cv2.getTrackbarPos('extended', 'result'))
        self.trees = cv2.getTrackbarPos('trees', 'result')
        self.checks = cv2.getTrackbarPos('checks', 'result')

        surfParameters = {
                'hessianThreshold': self.hessianThreshold,
                'nOctaves': self.nOctaves,
                'nOctaveLayers': self.nOctaveLayers,
                'extended': self.extended,
                'upright': False,
                }

        surf = cv2.xfeatures2d.SURF_create(**surfParameters)
        kp1, des1 = surf.detectAndCompute(self.queryImage, None)
        kp2, des2 = surf.detectAndCompute(self.trainingImage, None)
        des1, des2 = np.float32(des1), np.float32(des2)

        self.indexParams = dict(algorithm=self.FLANN_INDEX_KDTREE,
                                trees=self.trees)
        self.searchParams = dict(checks=self.checks)
        flann = cv2.FlannBasedMatcher(self.indexParams, self.searchParams)

        matches = flann.knnMatch(des1, des2, k=2)
        matchesMaks = [[0, 0] for i in xrange(len(matches))]
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMaks[i] = [1, 0]
        drawParams = dict(matchColor=(0, 255, 0),
                          singlePointColor=(255, 0, 0),
                          matchesMask=matchesMaks,
                          flags=0)

        resultImage = cv2.drawMatchesKnn(self.queryImage, kp1,
                                         self.trainingImage, kp2,
                                         matches, None,
                                         **drawParams)
        cv2.imshow('result', resultImage)


queryImage = cv2.imread('trainImages/kolj_ind/kolj_ind-0004.pgm', 0)
trainingImage = cv2.imread('trainImages/kolj_ind/kolj_ind-0004.pgm', 0)
trainingImage = cv2.resize(trainingImage, None, fx=1.2, fy=1.2,
                           interpolation=cv2.INTER_CUBIC)
"""
# create ORB and detect/compute
orbParameters = {'nfeatures': 1000,
                  'scaleFactor': 1.1,
                  'nlevels': 40,
                  'edgeThreshold': 5,  # roughly match the patchSize parameter
                  'firstLevel': 0,  # 0
                  'WTA_K': 4,  # 2, 3, 4
                  'scoreType': cv2.ORB_HARRIS_SCORE,  # cv2.ORB_FAST_SCORE
                  'patchSize': 5
                  }
orb = cv2.ORB_create(**orbParameters)
kp1, des1 = orb.detectAndCompute(queryImage, None)
kp2, des2 = orb.detectAndCompute(trainingImage, None)
des1, des2 = np.float32(des1), np.float32(des2)

# Create SIFT and detect/compute
#siftParameters = {'nfeatures': 500,
#                  'nOctaveLayers': 10,
#                  'contrastThreshold': 0.026,  #larger threshold,less features
#                  'edgeThreshold': 40,  # larger threshold, more features
#                  'sigma': 1.5  # weak camera, reduce the number
#                  }
#sift = cv2.xfeatures2d.SIFT_create(**siftParameters)
#kp1, des1 = sift.detectAndCompute(queryImage, None)
#kp2, des2 = sift.detectAndCompute(trainingImage, None)
#des1, des2 = np.float32(des1), np.float32(des2)

# Create SURF and detect/compute
#surfParameters = {'hessianThreshold': 400,  # larger value, less features
#                  # 300-500 good value
#                  'nOctaves': 25,  # number of pyramid octaves
#                  'nOctaveLayers': 10,  # number of layers within octave
#                  'extended': True,  # 128/64 elements descriptors
#                  'upright': False,  # compute orientation or not
#                  }
#surf = cv2.xfeatures2d.SURF_create(**surfParameters)
#kp1, des1 = surf.detectAndCompute(queryImage, None)
#kp2, des2 = surf.detectAndCompute(trainingImage, None)
#des1, des2 = np.float32(des1), np.float32(des2)

# FLANN matcher parameters for ORB detection
FLANN_INDEX_LSH = 0
indexParams = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12,
                   multi_probe_level=1)

# FLANN matcher parameters for SIFT/SURF detection
#FLANN_INDEX_KDTREE = 0
#indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)

# Search params and matching
searchParams = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
matches = flann.knnMatch(des1, des2, k=2)

# Prepare an empty mask to draw good matches
matchesMaks = [[0, 0] for i in xrange(len(matches))]

# David G. Lowe's ratio test, populate the mask
for i, (m, n) in enumerate(matches):
    if m.distance < 0.7 * n.distance:
        matchesMaks[i] = [1, 0]

drawParams = dict(matchColor=(0, 255, 0),
                  singlePointColor=(255, 0, 0),
                  matchesMask=matchesMaks,
                  flags=0)
resultImage = cv2.drawMatchesKnn(queryImage, kp1, trainingImage, kp2, matches,
                                 None, **drawParams)
"""
cv2.namedWindow('result', cv2.WINDOW_AUTOSIZE)

#fMatching = ORBWindow(queryImage, trainingImage)
#fMatching = SIFTWindow(queryImage, trainingImage)
fMatching = SURFWindow(queryImage, trainingImage)

cv2.waitKey()
cv2.destroyAllWindows()
