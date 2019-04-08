#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:36:06 2017

@author: tomislav
"""

import cv2
import numpy as np
import os
import sys


dataPath = 'trainImages/'
SAMPLES = 423


def path(cls, i):
    cls = cls + str(i+1).zfill(4) + '.pgm'
    return os.path.join(dataPath, cls)


def get_flann_matcher():
    FLANN_INDEX_KDTREE = 0
    indexParams = {'algorithm': FLANN_INDEX_KDTREE,
                   'trees': 5
                   }
    searchParams = {'checks': 50}
    return cv2.FlannBasedMatcher(indexParams, searchParams)


def get_bow_extractor(extractor, matcher):
    return cv2.BOWImgDescriptorExtractor(extractor, matcher)


def get_extract_detect():
    surfParameters = {
            'hessianThreshold': 550,  # larger value, less features
            # 300-500 good value
            'nOctaves': 2,  # number of pyramid octaves
            'nOctaveLayers': 20,  # number of layers within octave
            'extended': False,  # 128/64 elements descriptors
            'upright': False  # compute orientation or not
            }
    return (cv2.xfeatures2d.SURF_create(**surfParameters),
            cv2.xfeatures2d.SURF_create(**surfParameters))


def extract_des(fn, extractor, detector):
    im = cv2.imread(fn, 0)
    return extractor.compute(im, detector.detect(im))[1]


def bow_features(img, extractor_bow, detector):
    return extractor_bow.compute(img, detector.detect(img))


def obj_detector():
    cep = 'cep_ind/cep_ind-'
    koljeno = 'kolj_ind/kolj_ind-'
    detect, extract = get_extract_detect()
    matcher = get_flann_matcher()
    print 'Building BOWKMeansTrainer ...'
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(2048)
    extract_bow = get_bow_extractor(extract, matcher)

    print 'Adding features to trainer ...'
    for i in xrange(SAMPLES):
        print 'Adding image couple {}'.format(i+1)
        bow_kmeans_trainer.add(extract_des(path(cep, i), extract, detect))
        bow_kmeans_trainer.add(extract_des(path(koljeno, i), extract, detect))

    print 'Number of features in BOW: {}'.format(
            bow_kmeans_trainer.descriptorsCount())
    print 'Clustering features ...'
    voc = bow_kmeans_trainer.cluster()
    print 'Saving BOW vocabulary ...'
    fs = cv2.FileStorage('bow_vocabulary_ind.xml', cv2.FileStorage_WRITE)
    fs.write('bow-vocabulary-ind', voc)
    fs.release()
#    print 'Loading BOW vocabulary from file ...'
#    fs = cv2.FileStorage('bow_vocabulary.xml', cv2.FileStorage_READ)
#    node = fs.getNode('bow-vocabulary')
#    voc = node.mat()
#    fs.release()
    print 'Setting BOW vocabulary into extractor ...'
    extract_bow.setVocabulary(voc)

    traindata, trainlabels = [], []
    print 'Adding to train data'
    for i in xrange(SAMPLES):
        print 'Adding image couple {}'.format(i+1)
        traindata.extend(bow_features(cv2.imread(path(cep, i), 0),
                                      extract_bow, detect))
        trainlabels.append(1)
        traindata.extend(bow_features(cv2.imread(path(koljeno, i), 0),
                                      extract_bow, detect))
        trainlabels.append(2)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
#    svm.setGamma(2**-1)
#    svm.setNu(0.5)
    svm.setC(2**8)
    svm.setKernel(cv2.ml.SVM_INTER)

    print 'Training and saving SVN ...'
    svmTrainData = cv2.ml.TrainData_create(np.array(traindata),
                                           cv2.ml.ROW_SAMPLE,
                                           np.array(trainlabels))
    svm.train(svmTrainData)
    svm.save('svm_data_ind.xml')
    return svm, extract_bow
