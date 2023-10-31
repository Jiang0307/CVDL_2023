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

IMAGES = []
NX = 11
NY = 8

def draw_corner():
    winSize = (5, 5)
    zeroZone = (-1, -1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    temp = np.asarray(IMAGES)
    result = []
    for img in temp:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (NX,NY) , None)
        temp = cv2.cornerSubPix(gray, corners, winSize, zeroZone, criteria)
        result.append(cv2.drawChessboardCorners(gray, (NX,NY), corners, ret) )
        
    return result

def undistort():
    temp = np.asarray(IMAGES)
    objp = np.zeros((1, NX*NY, 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:NX, 0:NY].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    result = []
    for img in temp:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (NX , NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray , corners , (NX,NY) , (-1,-1) , criteria)
            imgpoints.append(corners2)
    ret, intrinsic_matrix , distortion_coefficients , rvecs , tvecs = cv2.calibrateCamera(objpoints , imgpoints , gray.shape[::-1] , None , None)
    temp = np.asarray(IMAGES)
    for img in temp:
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distortion_coefficients, (w,h), 1, (w,h))
        mapx, mapy = cv2.initUndistortRectifyMap(intrinsic_matrix, distortion_coefficients, None, newcameramtx, (w,h), 5)
        img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = roi
        img = img[y:y+h, x:x+w]
        result.append(img)
    return result

def draw_char(img, char_list:list):
    draw_image = img.copy()
    for line in char_list:
        line = line.reshape(2,2)
        # (0, 0, 255) 为蓝色线条，2 为线条宽度
        draw_image = cv2.polylines(draw_image, [line], isClosed=False, color=(0, 0 , 255), thickness=5)
        # draw_image = cv2.line(draw_image, tuple(line[0]), tuple(line[1]), (0,255,0), 10, cv2.LINE_AA)
    return draw_image


class MainWindow(QtWidgets.QMainWindow,UI):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.char_in_board = [ # coordinate for 6 charter in board (x, y) ==> (w, h)
            [7,5,0], # slot 1
            [4,5,0], # slot 2
            [1,5,0], # slot 3
            [7,2,0], # slot 4
            [4,2,0], # slot 5
            [1,2,0]  # slot 6
        ]


        self.setup_UI(self)
#=================================================Question 1=================================================
        self.load_foler_pushButton.clicked.connect(self.load_folder)
        self.find_corners_pushButton.clicked.connect(self.Q1_1)
        self.find_intrinsic_pushButton.clicked.connect(self.Q1_2)
        self.find_extrinsic_pushButton.clicked.connect(self.Q1_3)
        self.find_distortion_pushButton.clicked.connect(self.Q1_4)
        self.show_undistorted_result_pushButton.clicked.connect(self.Q1_5)

#=================================================Question 2=================================================
        self.show_words_on_board_pushButton.clicked.connect(self.Q2_1)
        self.show_words_vertical_pushButton.clicked.connect(self.Q2_2)

    def load_folder(self):
        IMAGES.clear()
        self.comboBox.clear()
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        files, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        self.image_count = 0

        for file_path in files:
            try:
                image = cv2.imread(file_path)
                IMAGES.append(image)
                print(f'Loaded image: {file_path}')
                self.image_count += 1
            except Exception as e:
                print(f'Error loading image: {e}')
        
        for i in range(self.image_count):
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
        self.images = np.asarray(IMAGES)
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
        print("Intrinsic : ")
        print(intrinsic_matrix)
        print("")
        return
    
    def Q1_3(self):
        print("\nprocessing...")
        extrinsic_matrices = []
        self.images = np.asarray(IMAGES)
        # 取得index
        index = int( self.comboBox.currentText() )
        if index<0 or index>14:
            return
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 , 0.001)
        objp = np.zeros((1 , NX*NY , 3) , np.float32)
        objp[0,:,:2] = np.mgrid[0:NX , 0:NY].T.reshape(-1,2)
        objpoints = []
        imgpoints = []

        # 求出intrinsic matrix
        for img in self.images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (NX,NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray , corners , (NX,NY) , (-1,-1) , criteria)
                imgpoints.append(corners2)       
        
        # 求出rvec[idnex]的extrinsic matrix
        ret, intrinsic_matrix , distortion_coefficients , rvecs , tvecs = cv2.calibrateCamera(objpoints , imgpoints , gray.shape[::-1] , None , None)
        r = np.zeros((3,3))
        cv2.Rodrigues(rvecs[index] , r , jacobian=0)
        extrinstic_matrix = np.concatenate((r , tvecs[index]) , axis=1)
        print("Extrinsic : ")
        print(extrinstic_matrix)
        print("")
        
        return

    def Q1_4(self):
        print("\nprocessing...")
        self.images = np.asarray(IMAGES)
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
        print("Distortion : ")
        print(intrinsic_matrix)
        print("")
        return

    def Q1_5(self):
        print("processing...")
        self.images = np.asarray(IMAGES)
        result_2_5 = undistort()

        title_1 = "Distorted"
        title_2 = "Undistorted"
        for i in range(self.image_count):
            distorted_img = self.images[i]
            distorted_img = cv2.resize(distorted_img , (800,800) )
            undistorted_img = result_2_5[i]
            undistorted_img = cv2.resize(undistorted_img , (800,800) )

            cv2.imshow(title_1 , distorted_img)
            cv2.imshow(title_2 , undistorted_img)
            cv2.waitKey(500)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return        

#=================================================Question 2=================================================
    def Q2_1(self):
        print("\nprocessing...")
        library_path = str( dataset_path.joinpath("Q2_Image").joinpath("Q2_lib").joinpath("alphabet_lib_onboard.yaml") )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 , 0.001)
        objp = np.zeros((1 , NX*NY , 3) , np.float32)
        objp[0,:,:2] = np.mgrid[0:NX , 0:NY].T.reshape(-1,2)
        objpoints = []
        imgpoints = []

        word = self.lineEdit.text()[:6].upper()

        for img in IMAGES:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (NX,NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray , corners , (NX, NY) , (-1,-1) , criteria)
                imgpoints.append(corners2)

        for index, image in enumerate(IMAGES):
            h, w = image.shape[:2]
            draw_image = image.copy()
            ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints , imgpoints , (w,h) , None, None)
            if ret:
                rvec = np.array(rvecs[index])
                tvec = np.array(tvecs[index]).reshape(3,1)
                for i_char, char in enumerate(word):
                    input  = cv2.FileStorage(library_path , cv2.FILE_STORAGE_READ)
                    char_matrix = np.float32( input.getNode(char).mat() )
                    print("char_matrix : ",char_matrix)
                    line_list = []
                    for eachline in char_matrix:
                        ach = np.float32([self.char_in_board[i_char], self.char_in_board[i_char]])
                        eachline = np.add(eachline, ach)
                        image_points, jac = cv2.projectPoints(eachline, rvec, tvec, intrinsic_mtx, dist)
                        line_list.append( np.int32(image_points) )
                    draw_image = draw_char(draw_image, line_list)
                cv2.namedWindow("WORD ON BOARD", cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow("WORD ON BOARD", draw_image)
                cv2.waitKey(1000)
        self.setEnabled(True)
    
#=================================================Question 3=================================================
    def Q2_2(self):
        print("\nprocessing...")
        library_path = str( dataset_path.joinpath("Q2_Image").joinpath("Q2_lib").joinpath("alphabet_lib_vertical.yaml") )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 30 , 0.001)
        objp = np.zeros((1 , NX*NY , 3) , np.float32)
        objp[0,:,:2] = np.mgrid[0:NX , 0:NY].T.reshape(-1,2)
        objpoints = []
        imgpoints = []

        word = self.lineEdit.text()[:6].upper()

        for img in IMAGES:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (NX,NY), cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray , corners , (NX, NY) , (-1,-1) , criteria)
                imgpoints.append(corners2)

        for index, image in enumerate(IMAGES):
            h, w = image.shape[:2]
            draw_image = image.copy()
            ret, intrinsic_mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints , imgpoints , (w,h) , None, None)
            if ret:
                rvec = np.array(rvecs[index])
                tvec = np.array(tvecs[index]).reshape(3,1)
                for i_char, char in enumerate(word):
                    input  = cv2.FileStorage(library_path , cv2.FILE_STORAGE_READ)
                    char_matrix = np.float32( input.getNode(char).mat() )
                    print("char_matrix : ",char_matrix)
                    line_list = []
                    for eachline in char_matrix:
                        ach = np.float32([self.char_in_board[i_char], self.char_in_board[i_char]])
                        eachline = np.add(eachline, ach)
                        image_points, jac = cv2.projectPoints(eachline, rvec, tvec, intrinsic_mtx, dist)
                        line_list.append( np.int32(image_points) )
                    draw_image = draw_char(draw_image, line_list)
                cv2.namedWindow("WORD ON BOARD", cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow("WORD ON BOARD", draw_image)
                cv2.waitKey(1000)
        self.setEnabled(True)




#=================================================Question 4=================================================






#=================================================Question 5=================================================





if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())