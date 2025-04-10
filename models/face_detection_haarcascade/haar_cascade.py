# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

from itertools import product

import numpy as np
import cv2 as cv

class HaarCascade:
    def __init__(self, modelPath, scale_factor=1.05, min_neighbors=3, min_size=(30, 30), max_size=(300, 300), backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        self._max_size = max_size

        self._model = cv.CascadeClassifier(self._modelPath)

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = cv.CascadeClassifier(self._modelPath)

    def setInputSize(self, input_size):
        pass

    def infer(self, image):

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # gray = cv.equalizeHist(gray)

        faces = self._model.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            # minSize=self._min_size,
            # maxSize=self._max_size,
        )

        if len(faces) == 0:
            return np.empty(shape=(0, 5))
        
        # Kiểm tra hình dạng của faces - nếu là 1 chiều, reshape nó
        if len(faces.shape) == 1:  # Trường hợp chỉ có một khuôn mặt được phát hiện
            faces = faces.reshape(1, -1)
        
        # Thêm cột score (mặc định là 1.0)
        # scores = np.ones((faces.shape[0], 1))
        scores = np.random.uniform(0.85, 0.99, (len(faces), 1))
        
        # Kết hợp faces với scores
        result = np.hstack((faces, scores))        
        return result
