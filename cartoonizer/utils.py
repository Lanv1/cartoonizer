import cv2 as cv
import numpy as np

class Utils:
    @staticmethod
    def resize_with_aspect_ration(image, width=None, height=None, inter=cv.INTER_AREA):
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv.resize(image, dim, interpolation=inter)