# Application of Machine Vision for Injection Moulding Products Detection and Recognition

Python scripts for image acquisition, feature extraction, SVM algorithm training and object detection and recognition.
Final result is object contour, position and orientation.
Research resulted in professional paper [(Bazina, Marković, Pavletić, & Jurković, 2017)](#references).

## Abstract

Machine Vision is found useful in industry, especially at facilities where measuring and quality control are of key importance. Also, where there is requirement for automatic robot manipulation, Machine Vision is very valuable. This paper presents an overview of fundamental concepts related to image processing, as a basis for the application of Machine Vision. Different methods of digital image analysis and detection or recognition of objects are presented in this work paper, together with their application examples, supported by theoretical elaboration. Python programming language and the OpenCV and NumPy program packages, were used for development of software solution for detection and recognition of products, manufactured by the plastic injection molding process. The efficiency of the software solution in object recognition, contour detection, position and orientation definition is tested on a series of images, obtained with an industrial camera.

## Content

Structure:
* `grayscale_resize.py`
  * Reducing image size, converting images to grayscale and saving in .pgm
    format
* `testImage_resize.py`
  * Resizing test images
* `FLANN_ORB_match.py`
  * GUI for comparing ORB, SIFT and SURF features
  * Matching features using FLANN
  * Sliders for parameter adjustment
* obj_detector:
  * `contour_detection.py`
    * GUI and sliders for contour detection algorithm parameters adjustment
    * bilateralFilter, Canny, morphologyEx, findContours, areaThresholding
  * `detector.py`
    * Extracting and descripting features from train images, K-means clustering, constructing and saving BOVW, training and saving SVM
  * `non_maximum.py`
    * NMS (non-maximum suppression) algorithm written in pure Python according to [Tombone's Computer Vision Blog](#references)
  * `pyramid.py`
    * Image pyramid algorithm
  * `sliding_window.py`
    * Sliding window technique
* `obj_SW_IP_NMS.py`
  * Testing on test images
* `managers.py`
  * Capturing video
* `bow_vocabulary_ind.xml`
  * saved BOVW vocabulary
* `svm_data_ind.xml`
  * saved and trained SVM algorithm
* trainImages:
  * Folder containing images of two different objects for SVM training
* testImages/test2:
  * Folder containing images for SVM testing

Parts of code were written according to [(Minichino & Howse, 2015)](#references)


### Prerequisites

Following Python modules were used:
* [OpenCV](https://github.com/opencv/opencv)
* [NumPy](https://github.com/numpy/numpy)

## References

* [Bazina, T., Marković, M., Pavletić, D., & Jurković, Z. (2017, January). Application of Machine Vision for Injection Moulding Products Detection and Recognition. In 7 International Conference Mechanical Technologies and Structural Materials 2017.](http://www.strojarska-tehnologija.hr/img/pdf/MTSM2017_CONFERENCE_PROCEEDINGS.compressed.pdf)
* [Tombone's Computer Vision Blog: blazing fast nms.m](http://www.computervisionblog.com/2011/08/blazing-fast-nmsm-from-exemplar-svm.html)
* Minichino, J., & Howse, J. (2015). Learning OpenCV 3 Computer Vision with Python - Second Edition. Packt Publishing.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
