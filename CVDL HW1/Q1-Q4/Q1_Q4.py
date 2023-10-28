from math import sqrt
from typing import Pattern
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
from UI_Q1_Q4 import UI

import os
import math
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

dataset_path = Path(__file__).parent.parent.joinpath("Dataset")

def cv2_imread(path):
    img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

Q1_IMAGES = []
NX = 11
NY = 8

def draw_corner():
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    temp = np.asarray(Q1_IMAGES)
    result = []
    for img in temp:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (NX,NY) , None)
        temp = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)
        result.append(cv2.drawChessboardCorners(gray, (NX,NY), corners, ret) )
        
    return result


class MainWindow(QtWidgets.QMainWindow,UI):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.setup_UI(self)
#=================================================Question 1=================================================
        self.load_foler_pushButton.clicked.connect(self.load_folder)
        # self.load_image_l_pushButton.clicked.connect(self.Color_Separetion)
        # self.load_image_r_pushButton.clicked.connect(self.Color_Transformation)

#=================================================Question 1=================================================
        self.find_corners_pushButton.clicked.connect(self.Q1_1)
        self.find_intrinsic_pushButton.clicked.connect(self.Q1_2)
        self.find_extrinsic_pushButton.clicked.connect(self.Q1_3)


    def load_folder(self):
        self.comboBox.clear()
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        files, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        count = 0

        for file_path in files:
            try:
                image = cv2.imread(file_path)
                Q1_IMAGES.append(image)
                print(f'Loaded image: {file_path}')
                count += 1
            except Exception as e:
                print(f'Error loading image: {e}')
        
        for i in range(count):
            print(i+1)
            self.comboBox.addItem(str(i+1))


    def Q1_1(self):
        title = "2-1"
        result_2_1 = draw_corner()
        for img in result_2_1:
            cv2.namedWindow(title , cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title , 1500 , 1500)
            cv2.imshow(title , img)
            cv2.waitKey(500)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return

    def Q1_2(self):
        print("\nprocessing...")
        self.images = np.asarray(Q1_IMAGES)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 , 0.001)
        objp = np.zeros((1 , NX*NY , 3) , np.float32)
        objp[0,:,:2] = np.mgrid[0:NX , 0:NY].T.reshape(-1,2)
        objpoints = []
        imgpoints = []
        for img in self.images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (NX,NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray , corners , (NX, NY) , (-1,-1) , criteria)
                imgpoints.append(corners2)       
        ret, intrinsic_matrix , distortion_coefficients , rvecs , tvecs = cv2.calibrateCamera(objpoints , imgpoints , gray.shape[::-1] , None , None)
        print("intrinsic matrix: ")
        print(intrinsic_matrix)
        print("")
        return
    
    def Q1_3(self):
        print("\nprocessing...")
        try:
            self.images = np.asarray(Q1_IMAGES)
            index = int( self.LineEdit2_3.text() )
            if index<0 or index>14:
                return
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 , 0.001)
            objp = np.zeros((1 , NX*NY , 3) , np.float32)
            objp[0,:,:2] = np.mgrid[0:NX , 0:NY].T.reshape(-1,2)
            objpoints = []
            imgpoints = []
            
            for img in self.images:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (NX , NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret == True:
                    objpoints.append(objp)
                    corners2 = cv2.cornerSubPix(gray , corners , (NX,NY) , (-1,-1) , criteria)
                    imgpoints.append(corners2)       
            ret, intrinsic_matrix , distortion_coefficients , rvecs , tvecs = cv2.calibrateCamera(objpoints , imgpoints , gray.shape[::-1] , None , None)
            r = np.zeros((3,3))
            cv2.Rodrigues(rvecs[index] , r , jacobian=0)
            extrinstic_matrix = np.concatenate((r , tvecs[index]) , axis=1)
            print("extrinsic matrix: ")
            print(extrinstic_matrix)
            print("")
        except ValueError:
            pass
        except Exception:
            pass
        return


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())