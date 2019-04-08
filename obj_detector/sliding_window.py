#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:06:15 2017

@author: tomislav
"""


def sliding_window(image, stepSize=20, windowSize=(200, 200)):
    for y in xrange(0, image.shape[0]-windowSize[1]+stepSize, stepSize):
        for x in xrange(0, image.shape[1]-windowSize[0]+stepSize, stepSize):
            yield (x, y, image[y:y+windowSize[1], x:x+windowSize[0]])
