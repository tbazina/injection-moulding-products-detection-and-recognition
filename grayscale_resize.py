#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 01:40:23 2017

@author: tomislav
"""

import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Directory with full size images
directory = 'trainImages/kolj_ind'
files = [join(directory, f) for f in listdir(directory) if
         isfile(join(directory, f))]

# Name of output images
img_name = join(directory, 'kolj_ind-')
num = 1
for path_img in files:
    img = cv2.imread(path_img, cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, dsize=(256, 179), dst=None,  # fx=0.1, fy=0.1,
                         interpolation=cv2.INTER_AREA)
    full_name = img_name + str(num).zfill(4) + '.pgm'
    cv2.imwrite(full_name, resized)
    print path_img + '-->' + full_name
    num += 1
