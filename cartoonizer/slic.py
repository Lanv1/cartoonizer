from pprint import pprint

import cv2 as cv
import numpy as np

from utils import Utils


class Slic:
    def __init__(self, img, region_size, compactness):
        self.__img = img
        self.__compactness = compactness
        self.__region_size = region_size
        self.__slic = None
        self.__img_slic = None
        self.__superpixels = None
        self.__img_slic_avg = None

    def compute_slic(self):
        self.__slic = cv.ximgproc.createSuperpixelSLIC(self.__img, algorithm=cv.ximgproc.SLIC, region_size=self.__region_size, ruler=self.__compactness)
        self.__slic.iterate()

        # retrieve the segmentation result
        self.__img_slic = self.__slic.getLabels()  # height x width matrix. Each component indicates the superpixel index of the corresponding pixel position

    def rearrange_to_superpixels(self):
        nb_sp = self.__slic.getNumberOfSuperpixels()
        self.__superpixels = [[] for _ in range(nb_sp)]
        nb_line = range(len(self.__img_slic))
        for x in nb_line:
            nb_col = range(len(self.__img_slic[x]))
            for y in nb_col:
                self.__superpixels[self.__img_slic[x, y]].append([x, y])

    def compute_average_superpixels(self):
        nb_sp = self.__slic.getNumberOfSuperpixels()
        sp_avg = [{"r": 0, "g": 0, "b": 0} for _ in range(nb_sp)]
        nb_px_sp = [0 for _ in range(nb_sp)]
        for i_sp in range(nb_sp):
            for x, y in self.__superpixels[i_sp]:
                b, g, r = (self.__img[x, y])
                nb_px_sp[i_sp] += 1
                sp_avg[i_sp]["r"] += r
                sp_avg[i_sp]["g"] += g
                sp_avg[i_sp]["b"] += b

            sp_avg[i_sp]["r"] /= nb_px_sp[i_sp]
            sp_avg[i_sp]["g"] /= nb_px_sp[i_sp]
            sp_avg[i_sp]["b"] /= nb_px_sp[i_sp]

        self.__img_slic_avg = self.__img.copy()
        for i_sp in range(nb_sp):
            for x, y in self.__superpixels[i_sp]:
                self.__img_slic_avg[x, y] = (sp_avg[i_sp]["b"], sp_avg[i_sp]["g"], sp_avg[i_sp]["r"])

    def get_superpixels(self):
        return self.__superpixels

    def get_img_slic_avg(self):
        return self.__img_slic_avg
