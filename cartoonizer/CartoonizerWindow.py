import os
import sys
from functools import partial

import cv2 as cv
import numpy as np
from PySide2 import QtCore
from PySide2 import QtGui
from PySide2 import QtWidgets
from numpy.core.defchararray import isnumeric

from slic import Slic


class CartoonizerWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()

        # Model attributes
        self.__filename = 'asset/faker.jpg'
        self.__img_base = cv.imread(self.__filename)
        self.__img_display = self.__img_base.copy()
        self.__img_slic = self.__img_base.copy()
        self.__img_kmean = self.__img_base.copy()
        self.__region_size = 50
        self.__compactness = 0
        self.__blur_kernel_size = 5
        self.__dilat_kernel_size = 1
        self.__edge_t1 = 100
        self.__edge_t2 = 200
        self.__k = 10

        # name the window
        self.setWindowTitle("Cartoonizer")
        # make the window a "tool" in Maya's eyes so that it stays on top when you click off
        self.setWindowFlags(QtCore.Qt.Tool)
        # Makes the object get deleted from memory, not just hidden, when it is closed.
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)

        # Create the layout, linking it to actions and refresh the display
        self.__create_layout()
        self.__status_bar()
        self.__actions()
        self.__menu()
        self.__context_menu()
        self.__link_actions()

        cv.imshow("Cartoonizer", self.__img_display)

    def __status_bar(self):
        self.setStatusBar(QtWidgets.QStatusBar(self))

    def __actions(self):
        self.act_open_file = QtWidgets.QAction("Open", self)
        self.act_open_file.setStatusTip("Open a new file")
        self.act_open_file.setShortcut("Ctrl+O")
        self.act_open_file.triggered.connect(self.__open_file)

        self.act_save = QtWidgets.QAction("Save", self)
        self.act_save.setStatusTip("Save the file")
        self.act_save.triggered.connect(self.__save_file)

        self.act_save_as = QtWidgets.QAction("Save as", self)
        self.act_save_as.setStatusTip("Save as a new file")
        self.act_save_as.setShortcut("Ctrl+S")
        self.act_save_as.triggered.connect(self.__save_file_as)

    def __menu(self):
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction(self.act_open_file)
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_save_as)

    def __context_menu(self):
        central_widget = self.centralWidget()
        central_widget.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        central_widget.addAction(self.act_open_file)
        central_widget.addAction(self.act_save)
        central_widget.addAction(self.act_save_as)

    def __set_new_file(self, filename):
        if filename is not None:
            self.__filename = filename
            self.__img_base = cv.imread('asset/faker.jpg')

    def __save_file(self):
        self.statusBar().showMessage("File saved", 1000)
        cv.imwrite(self.file_name[0], self.__img_display)

    def __save_file_as(self):
        self.statusBar().showMessage("Save as a new file", 1000)
        self.file_name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', self.__filename,
                                                          "Image files (*.pgm *.ppm *.jpg *.png *.tif)")
        cv.imwrite(self.file_name[0], self.__img_display)

    def __open_file(self):
        self.statusBar().showMessage("Open a new file", 1000)

        self.__set_new_file(QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', "",
                                                                  "Image files (*.pgm *.ppm *.jpg *.png *.tif)")[0])

    # Create the layout
    def __create_layout(self):
        # self.setFixedSize(720, 780)
        # self.move(QtWidgets.QDesktopWidget().availableGeometry().center() - self.frameGeometry().center())

        # Some aesthetic value
        size_btn = QtCore.QSize(200, 30)
        size_slider = QtCore.QSize(400, 30)

        # Main Layout
        main_layout = QtWidgets.QVBoxLayout()
        self.__central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.__central_widget)
        self.__central_widget.setLayout(main_layout)

        # Layout top
        layout_top = QtWidgets.QHBoxLayout()
        layout_top.setContentsMargins(15,15,15,15)
        main_layout.addLayout(layout_top)

        # Layout slic
        layout_slic = QtWidgets.QVBoxLayout()
        layout_top.addLayout(layout_slic)
        self.__ui_btn_submit_slic = QtWidgets.QPushButton("Compute SLIC")
        self.__ui_btn_submit_slic.setFixedSize(size_btn)
        self.__ui_region_size = QtWidgets.QLineEdit(str(self.__region_size))
        self.__ui_compactness = QtWidgets.QLineEdit(str(self.__compactness))
        layout_rs_param = QtWidgets.QHBoxLayout()
        layout_rs_param.addWidget(QtWidgets.QLabel("Region Size"), 0, QtCore.Qt.AlignRight)
        layout_rs_param.addWidget(self.__ui_region_size, 0, QtCore.Qt.AlignHCenter)
        layout_comp_param = QtWidgets.QHBoxLayout()
        layout_comp_param.addWidget(QtWidgets.QLabel("Compactness"), 0, QtCore.Qt.AlignRight)
        layout_comp_param.addWidget(self.__ui_compactness, 0, QtCore.Qt.AlignHCenter)
        layout_slic.addLayout(layout_rs_param)
        layout_slic.addLayout(layout_comp_param)
        layout_slic.addWidget(self.__ui_btn_submit_slic, 0, QtCore.Qt.AlignHCenter)

        # Layout kmean
        layout_kmean = QtWidgets.QVBoxLayout()
        layout_top.addLayout(layout_kmean)
        self.__ui_btn_submit_kmean = QtWidgets.QPushButton("Compute KMEAN")
        self.__ui_btn_submit_kmean.setFixedSize(size_btn)
        self.__ui_k = QtWidgets.QLineEdit(str(self.__k))
        layout_k_param = QtWidgets.QHBoxLayout()
        layout_k_param.addWidget(QtWidgets.QLabel("Nb Cluster"), 0, QtCore.Qt.AlignRight)
        layout_k_param.addWidget(self.__ui_k, 0, QtCore.Qt.AlignHCenter)
        layout_kmean.addLayout(layout_k_param)
        layout_kmean.addWidget(self.__ui_btn_submit_kmean, 0, QtCore.Qt.AlignHCenter |QtCore.Qt.AlignBottom)

        # Layout bottom
        layout_bot = QtWidgets.QVBoxLayout()
        layout_bot.setContentsMargins(15,15,15,15)
        main_layout.addLayout(layout_bot)

        self.__ui_blur_kernel_size = QtWidgets.QLineEdit(str(self.__blur_kernel_size))
        self.__ui_dilat_kernel_size = QtWidgets.QLineEdit(str(self.__dilat_kernel_size))
        layout_blur_param = QtWidgets.QHBoxLayout()
        layout_blur_param.addWidget(QtWidgets.QLabel("Blur Kernel SIze"), 0, QtCore.Qt.AlignRight)
        layout_blur_param.addWidget(self.__ui_blur_kernel_size, 0, QtCore.Qt.AlignHCenter)
        layout_dilat_param = QtWidgets.QHBoxLayout()
        layout_dilat_param.addWidget(QtWidgets.QLabel("Dilatation Kernel Size"), 0, QtCore.Qt.AlignRight)
        layout_dilat_param.addWidget(self.__ui_dilat_kernel_size, 0, QtCore.Qt.AlignHCenter)
        layout_bot.addLayout(layout_blur_param)
        layout_bot.addLayout(layout_dilat_param)
        self.__ui_slider_t1 = QtWidgets.QSlider()
        self.__ui_slider_t2 = QtWidgets.QSlider()
        self.__ui_slider_t1.setOrientation(QtCore.Qt.Horizontal)
        self.__ui_slider_t1.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.__ui_slider_t1.setTickInterval(10)
        self.__ui_slider_t1.setMinimum(0)
        self.__ui_slider_t1.setMaximum(255)
        self.__ui_slider_t1.setFixedSize(size_slider)
        self.__ui_slider_t1.setValue(self.__edge_t1)
        self.__ui_slider_t2.setOrientation(QtCore.Qt.Horizontal)
        self.__ui_slider_t2.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.__ui_slider_t2.setTickInterval(10)
        self.__ui_slider_t2.setMinimum(0)
        self.__ui_slider_t2.setMaximum(255)
        self.__ui_slider_t2.setFixedSize(size_slider)
        self.__ui_slider_t2.setValue(self.__edge_t2)
        layout_t1_param = QtWidgets.QVBoxLayout()
        layout_t1_param.setContentsMargins(8,8,8,8)
        layout_t2_param = QtWidgets.QVBoxLayout()
        layout_t2_param.setContentsMargins(8,8,8,8)
        layout_t1_header_param = QtWidgets.QHBoxLayout()
        layout_t2_header_param = QtWidgets.QHBoxLayout()
        layout_t1_header_param.addWidget(QtWidgets.QLabel("Threshold Hysteresis Edges 1"), 0, QtCore.Qt.AlignLeft)
        self.__ui_value_t1 = QtWidgets.QLabel(str(self.__edge_t1))
        layout_t1_header_param.addWidget(self.__ui_value_t1, 0, QtCore.Qt.AlignRight)
        layout_t1_param.addLayout(layout_t1_header_param)
        layout_t1_param.addWidget(self.__ui_slider_t1, 0, QtCore.Qt.AlignLeft)
        layout_t2_header_param.addWidget(QtWidgets.QLabel("Threshold Hysteresis Edges 2"), 0, QtCore.Qt.AlignLeft)
        self.__ui_value_t2 = QtWidgets.QLabel(str(self.__edge_t2))
        layout_t2_header_param.addWidget(self.__ui_value_t2, 0, QtCore.Qt.AlignRight)
        layout_t2_param.addLayout(layout_t2_header_param)
        layout_t2_param.addWidget(self.__ui_slider_t2, 0, QtCore.Qt.AlignLeft)
        layout_bot.addLayout(layout_t1_param)
        layout_bot.addLayout(layout_t2_param)

    def __submit_slic(self):
        self.__compactness = int(self.__ui_compactness.text())
        self.__region_size = int(self.__ui_region_size.text())
        self.__compute_slic()
        self.__refresh_display()

    def __submit_kmean(self):
        self.__k = int(self.__ui_k.text())
        self.__compute_kmean()
        self.__refresh_display()

    def __compute_kmean(self):
        img = self.__img_base.copy()
        pixel_values = img.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, (centers) = cv.kmeans(pixel_values, self.__k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        labels = labels.flatten()
        res = centers[labels.flatten()]
        self.__img_slic = res.reshape(img.shape)


    def __compute_slic(self):
        slic = Slic(self.__img_base, self.__region_size, self.__compactness)
        slic.compute_slic()
        slic.rearrange_to_superpixels()
        slic.compute_average_superpixels()
        self.__img_slic = slic.get_img_slic_avg()

    def __blur_kernel_size_changed(self, value):
        if not isnumeric(value):
            self.__blur_kernel_size = 1
        else:
            value = int(value)
            if value == 0:
                self.__blur_kernel_size = 1
                self.__ui_blur_kernel_size.setText("1")
            elif value%2 == 0:
                self.__blur_kernel_size = value -1
            else:
                self.__blur_kernel_size = value

        self.__refresh_display()

    def __dilat_kernel_size_changed(self, value):
        if not isnumeric(value):
            self.__dilat_kernel_size = 1
        else:
            value = int(value)
            if value == 0:
                self.__dilat_kernel_size = 1
                self.__ui_dilat_kernel_size.setText("1")
            else:
                self.__dilat_kernel_size = value
        self.__refresh_display()

    def __edge_t1_changed(self, value):
        self.__edge_t1 = value
        self.__ui_value_t1.setText(str(value))
        self.__refresh_display()

    def __edge_t2_changed(self, value):
        self.__edge_t2 = value
        self.__ui_value_t2.setText(str(value))
        self.__refresh_display()

    # Link action to elements in the UI
    def __link_actions(self):
        self.__ui_btn_submit_slic.clicked.connect(self.__submit_slic)
        self.__ui_btn_submit_kmean.clicked.connect(self.__submit_kmean)
        self.__ui_blur_kernel_size.textChanged.connect(self.__blur_kernel_size_changed)
        self.__ui_dilat_kernel_size.textChanged.connect(self.__dilat_kernel_size_changed)
        self.__ui_slider_t1.valueChanged.connect(self.__edge_t1_changed)
        self.__ui_slider_t2.valueChanged.connect(self.__edge_t2_changed)

    def __refresh_display(self):
        img_blur = cv.GaussianBlur(self.__img_base, (self.__blur_kernel_size,self.__blur_kernel_size), 0)
        edges = cv.Canny(img_blur, threshold1=self.__edge_t1, threshold2=self.__edge_t2)

        kernel = np.ones((self.__dilat_kernel_size,self.__dilat_kernel_size), np.uint8)
        outline = cv.dilate(edges, kernel, iterations=1)

        height, width, channels = self.__img_base.shape
        color_img = np.zeros((height, width, 3), np.uint8)
        color_img[:] = (0, 0, 0)
        mask_inv = cv.bitwise_not(outline)
        result_bg = cv.bitwise_and(self.__img_slic, self.__img_slic, mask=mask_inv)
        result_fg = cv.bitwise_and(color_img, color_img, mask=outline)
        self.__img_display = cv.add(result_bg, result_fg)
        cv.imshow("Cartoonizer", self.__img_display)

