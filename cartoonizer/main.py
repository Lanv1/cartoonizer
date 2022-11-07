import sys

from PySide2 import QtWidgets

from CartoonizerWindow import CartoonizerWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ltp = CartoonizerWindow()
    ltp.show()
    app.exec_()