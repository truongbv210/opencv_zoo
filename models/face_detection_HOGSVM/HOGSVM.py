import cv2
import dlib
import numpy as np

class HOGSVM:
    def __init__(self, inputSize=[320, 320], backendId=0, targetId=0):
        self._inputSize = tuple(inputSize) # [w, h]
        self._backendId = backendId
        self._targetId = targetId

        self._model = dlib.get_frontal_face_detector()

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        # self._model = dlib.get_frontal_face_detector()

    def setInputSize(self, input_size):
        pass

    def infer(self, image):
        # Forward
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        rectangles, scores, _ = self._model.run(image_rgb, 1)
        if not rectangles:
            return np.empty(shape= (0, 5), dtype= np.float32)

        detections = []
        for rect, score in zip(rectangles, scores):
            x = max(0, rect.left())
            y = max(0, rect.top())
            w = rect.right() - x
            h = rect.bottom() - y

            detections.append([x, y, w, h, float(score)])

        return np.array(detections, dtype= np.float32)