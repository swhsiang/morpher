#!/usr/bin/env python3.7

#######################################################
#   Author:     Wen-Hsiang Shih
#   email:      wshih@purdue.edu
#   ID:         ee364d24
#   Date:       04/20/2019
#######################################################
"""
Creating a PyQt Application:

1- Create a UI file using the QtDesigner.

2- Convert the UI file to a Python file using the conversion tool:
    /package/eda/anaconda3/bin/pyuic5 <fileName.ui> -o <fileName.py>
   The generated file must NOT be modified, as indicated in the header warning!
   
3- Use the given file <blank.py> to create a consumer Python file, and write the code that drives the UI.

"""

# Import PyQt5 classes
import sys, imageio, logging, os
from Morphing import Morpher, ColorMorpher, Triangle, generate_tris
from PIL import ImageQt as pImageQt, Image as pImage
import numpy as np

from MorphingGUI import *
from PyQt5.QtGui import QImage, QPixmap, QBrush, QPen, QMouseEvent, QKeyEvent
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QGraphicsScene
from PyQt5.QtWidgets import QGraphicsView, QGraphicsPixmapItem, QWidget
from PyQt5.QtWidgets import QGraphicsSceneMouseEvent
from enum import IntEnum

import pysnooper


class State(IntEnum):
    Loading = 1
    Selecting = 2


class QScene(QGraphicsScene):
    img_points: np.ndarray
    point_file: str
    temp_point_and_ellipse: list
    img_tris: list
    pixmapItem: QGraphicsPixmapItem
    lineItems: list
    img: np.ndarray
    state: State

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # data storage
        self.img_tris = []
        self.img_points = np.array([], dtype=np.float64)
        self.temp_point_and_ellipse = []
        self.pixmapItem = None
        self.lineItems = []
        self.img = np.array([])
        self.point_file = ""
        self.state = State.Loading

    def savePoints(self):
        np.savetxt(self.point_file, self.img_points, fmt='%1.5f')

    def persitePoint(self):
        brush = QBrush(QtCore.Qt.blue)
        pen = QPen(QtCore.Qt.blue)

        point, ellipse = self.temp_point_and_ellipse[0]

        ellipse.setBrush(brush)
        ellipse.setPen(pen)

        temp = self.img_points.tolist()
        temp.append([point[0], point[1]])
        self.img_points = np.array(temp)
        self.savePoints()
        self.temp_point_and_ellipse.clear()

    def isSelectionMode(self) -> bool:
        return self.state == State.Selecting

    def clearSelectedPoint(self):
        _, ellipse = self.temp_point_and_ellipse[0]
        self.removeItem(ellipse)
        self.temp_point_and_ellipse.clear()

    def hasPointSelected(self):
        return len(self.temp_point_and_ellipse) > 0

    def selectPoint(self, point: "tuple(int, int)"):
        if not self.parent().point_selection_acl(self):
            return
        if not self.hasPointSelected() and self.isSelectionMode():
            radius = 16
            brush = QBrush(QtCore.Qt.green)
            pen = QPen(QtCore.Qt.green)
            ellipseItem = self.addEllipse(point[0], point[1], radius, radius,
                                          pen, brush)
            self.temp_point_and_ellipse.append((point, ellipseItem))

    def mouseReleaseEvent(self, event: QtCore.QEvent):
        self.parent().pairing(self)

        if isinstance(event, QGraphicsSceneMouseEvent):
            po = event.buttonDownScenePos(event.button())
            self.selectPoint((po.x(), po.y()))

    def keyReleaseEvent(self, event: QKeyEvent):
        if isinstance(event, QKeyEvent):
            if event.key() == QtCore.Qt.Key_Delete or event.key(
            ) == QtCore.Qt.Key_Backspace:
                if self.hasPointSelected():
                    self.clearSelectedPoint()

    def loadFiles(self, parent: QWidget) -> bool:
        filePath, _ = QFileDialog.getOpenFileName(
            parent,
            caption='Open image file ...',
            filter="Images (*.png *.jpg)")
        if not filePath:
            return False

        self.loadImg(filePath=filePath)
        self.point_file = f'{filePath}.txt'
        self.loadPoints(self.point_file)
        return True

    def loadImg(self, **kwargs) -> bool:
        if "img" in kwargs:
            try:
                self.img = kwargs.get("img")
                pixmap = QPixmap.fromImage(
                    pImageQt.ImageQt(pImage.fromarray(self.img)))
                self.clear()
                self.pixmapItem = self.addPixmap(pixmap)
            except Exception as e:
                logging.error(f"loadImg:img {e}")
                return False

        if "filePath" in kwargs and os.path.isfile(kwargs.get("filePath")):
            try:
                self.img = imageio.imread(kwargs.get("filePath"))
                pixmap = QPixmap.fromImage(QImage(kwargs.get("filePath")))
                self.clear()
                self.pixmapItem = self.addPixmap(pixmap)

            except Exception as e:
                logging.error(f"loadImg:filePath {e}")
                return False

        self.state = State.Selecting
        self.update()
        return True

    def img_loaded(self) -> bool:
        return self.img_loaded

    def loadPoints(self, filePath: str) -> bool:
        file_exist = True
        try:
            if os.path.isfile(filePath):
                self.img_points = np.loadtxt(
                    filePath, dtype=np.float64, ndmin=2)
            else:
                file_exist = False
        except Exception as e:
            logging.error(e)
            file_exist = False
        return file_exist

    def showPoints(self):
        radius = 12
        brush = QBrush(QtCore.Qt.red)
        pen = QPen(QtCore.Qt.red)

        # https://stackoverflow.com/questions/46382141/pyqt-mouse-events-in-qgraphicsview
        for coordinates in self.img_points:
            self.addEllipse(coordinates[0], coordinates[1], radius, radius,
                            pen, brush)

    @classmethod
    def showTriangle(self, l_scene: 'QScene', r_scene: 'QScene',
                     display: bool):
        for item in r_scene.lineItems:
            temp = r_scene.removeItem(item)
            del temp
        r_scene.lineItems.clear()

        for item in l_scene.lineItems:
            temp = l_scene.removeItem(item)
            del temp
        l_scene.lineItems.clear()

        if display is False:
            return

        l_points = l_scene.img_points
        r_points = r_scene.img_points
        (l_tris, r_tris) = generate_tris(l_points, r_points)

        pen = QPen(QtCore.Qt.blue)

        for tri in r_tris:
            vertices = tri.vertices
            r_scene.lineItems.extend([
                r_scene.addLine(vertices[0, 0], vertices[0, 1], vertices[1, 0],
                                vertices[1, 1], pen),
                r_scene.addLine(vertices[1, 0], vertices[1, 1], vertices[2, 0],
                                vertices[2, 1], pen),
                r_scene.addLine(vertices[0, 0], vertices[0, 1], vertices[2, 0],
                                vertices[2, 1], pen)
            ])

        for tri in l_tris:
            vertices = tri.vertices
            l_scene.lineItems.extend([
                l_scene.addLine(vertices[0, 0], vertices[0, 1], vertices[1, 0],
                                vertices[1, 1], pen),
                l_scene.addLine(vertices[1, 0], vertices[1, 1], vertices[2, 0],
                                vertices[2, 1], pen),
                l_scene.addLine(vertices[0, 0], vertices[0, 1], vertices[2, 0],
                                vertices[2, 1], pen)
            ])


class MorphingApp(QMainWindow, Ui_MainWindow):
    start_img_scene: QScene
    end_img_scene: QScene
    blend_img_scene: QScene

    def __init__(self, parent=None):
        super(MorphingApp, self).__init__(parent)
        self.setupUi(self)

        # disable bar & blend button
        self.BlendingRateBar.setEnabled(False)
        self.TriDisplayBox.setEnabled(False)
        self.BlendButton.setEnabled(False)
        self.alphaBox.setEnabled(False)

        # Event
        self.TriDisplayBox.clicked.connect(self.showDelaunayTriangles)
        self.alphaBox.setText("0.00")
        self.BlendingRateBar.valueChanged.connect(self.updateAlphaBox)
        self.start_img_scene = QScene(parent=self)
        self.StartImgCanvas.setScene(self.start_img_scene)

        self.end_img_scene = QScene(parent=self)
        self.EndImgCanvas.setScene(self.end_img_scene)

        self.blend_img_scene = QScene(parent=self)
        self.BlendedImgCanvas.setScene(self.blend_img_scene)

        # enable load img
        self.LoadEndImgBtn.setEnabled(True)
        self.LoadStartImgBtn.setEnabled(True)

        # Event
        self.LoadEndImgBtn.clicked.connect(self.loadEndImg)
        self.LoadStartImgBtn.clicked.connect(self.loadStartImg)

        self.BlendButton.clicked.connect(self.blending)

    def hasAlpha(self) -> bool:
        if not self.alphaBox.toPlainText().replace('.', '',
                                                   1).strip().isdigit():
            return False

        if self.getAlpha() > 1 or self.getAlpha() < 0:
            return False
        return True

    def getAlpha(self) -> float:
        return float(self.alphaBox.toPlainText().strip())

    def blending(self):
        l_points = self.start_img_scene.img_points
        r_points = self.end_img_scene.img_points
        (l_tris, r_tris) = generate_tris(l_points, r_points)
        self.start_img_scene.img_tris = l_tris
        self.end_img_scene.img_tris = r_tris

        if not self.hasAlpha():
            logging.error(f"invalid alpha {self.alphaBox.toPlainText()}")
            return

        alpha = self.getAlpha()
        if len(self.start_img_scene.img.shape) != len(
                self.end_img_scene.img.shape):
            raise Exception("size of two images mismatch!")

        if len(self.start_img_scene.img.shape) == 3:

            m = ColorMorpher(
                self.start_img_scene.img, self.start_img_scene.img_tris,
                self.end_img_scene.img, self.end_img_scene.img_tris)

        elif len(self.start_img_scene.img.shape) == 2:
            m = Morpher(self.start_img_scene.img,
                        self.start_img_scene.img_tris, self.end_img_scene.img,
                        self.end_img_scene.img_tris)
        else:
            raise Exception("size of two images is incorrect (channel)")

        res = m.getImageAtAlpha(alpha)
        self.blend_img_scene.loadImg(img=res)
        self.BlendedImgCanvas.fitInView(self.blend_img_scene.sceneRect())

    def loadStartImg(self):
        if self.start_img_scene.loadFiles(self):
            self.StartImgCanvas.fitInView(
                self.start_img_scene.pixmapItem.boundingRect())
            self.start_img_scene.showPoints()
        self.checkImgLoading()

    def loadEndImg(self):
        if self.end_img_scene.loadFiles(self):
            self.EndImgCanvas.fitInView(
                self.end_img_scene.pixmapItem.boundingRect())
            self.end_img_scene.showPoints()
        self.checkImgLoading()

    def checkImgLoading(self):
        if self.end_img_scene.isSelectionMode(
        ) and self.start_img_scene.isSelectionMode():
            self.BlendingRateBar.setEnabled(True)
            self.TriDisplayBox.setEnabled(True)
            self.BlendButton.setEnabled(True)
            self.alphaBox.setEnabled(True)

    def showDelaunayTriangles(self):
        QScene.showTriangle(self.start_img_scene, self.end_img_scene,
                            self.TriDisplayBox.isChecked())

    def mouseReleaseEvent(self, event: QtCore.QEvent):
        if isinstance(event, QMouseEvent):
            self.pairing()

    def updateAlphaBox(self):
        if self.hasAlpha():
            val = round((self.BlendingRateBar.sliderPosition() // 5) * 0.05, 2)
            self.alphaBox.setText(str(val))

    def pairing(self, source=None):
        if source is not None and source is not self.start_img_scene:
            return

        if self.start_img_scene.isSelectionMode() and\
                self.start_img_scene.hasPointSelected() and\
                    self.end_img_scene.isSelectionMode() and\
                        self.end_img_scene.hasPointSelected():
            self.start_img_scene.persitePoint()
            self.end_img_scene.persitePoint()

    def point_selection_acl(self, source: QScene):
        if isinstance(source, QScene) and (source is self.start_img_scene
                                           or source is self.end_img_scene):
            return True

        return False


if __name__ == "__main__":
    currentApp = QApplication(sys.argv)
    currentForm = MorphingApp()

    currentForm.show()
    currentApp.exec_()