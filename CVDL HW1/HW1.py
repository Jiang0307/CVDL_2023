from math import sqrt
from typing import Pattern
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog
from PySide6.QtGui import QPixmap, QImage, QImageReader

from HW1_UI import UI

import os
import math
import sys
import cv2
import glob
import matplotlib
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch.nn as nn

from scipy import signal
from pathlib import Path
from PIL import Image
from torchsummary import summary
from torchvision.models import vgg19_bn

matplotlib.use('TkAgg')
dataset_path = Path(__file__).parent.joinpath("Dataset")
training_path = Path(__file__).parent.joinpath("Training")

IMAGES = []
NX = 11
NY = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_dict = {0:"airplane" , 1:"automobile" , 2:"bird" , 3:"cat" , 4:"deer" , 5:"dog" , 6:"frog" , 7:"horse" , 8:"ship" , 9:"truck"}

def cv2_imread(path):
    img = cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def plot_to_image (fig):
    fig.canvas.draw()
    width,height = fig.canvas.get_width_height()
    buffer = np.frombuffer( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buffer.shape = (width,height,4)
    buffer = np.roll (buffer,3,axis=2)
    return buffer

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

def match_keypoints(img1, img2):
    sift = cv2.SIFT_create()
    grayimg1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    grayimg2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    kp1,des1 = sift.detectAndCompute(img1,None)
    kp2,des2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher.create()
    matches = bf.knnMatch(des1,des2,k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.75*n.distance:
            matchesMask[i] = [1, 0]
    drawpara = dict(singlePointColor=(0, 255, 0), matchesMask=matchesMask, flags=2)
    return cv2.drawMatchesKnn(grayimg1, kp1, grayimg2, kp2, matches, None, **drawpara)

    
def resizeImage(img):
    height = 400
    h,w = img.shape[:2]
    width = int (height * (w/h))
    return cv2.resize(img,(width,height))

class VGG19BN(nn.Module):
    def __init__(self):
        super(VGG19BN, self).__init__()
        self.vgg19_bn = torchvision.models.vgg19_bn(num_classes=10)
        self.features = self.vgg19_bn.features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # 减少了第一个全连接层的单元数
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Dropout(),
            nn.Linear(512, 10)  # 减少了第二个全连接层的单元数
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1), nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class MainWindow(QtWidgets.QMainWindow,UI):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.image = None
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

#=================================================Question 3=================================================
        self.load_image_l_pushButton.clicked.connect(self.load_image_l)
        self.load_image_r_pushButton.clicked.connect(self.load_image_r)
        self.stereo_disparity_map_pushButton.clicked.connect(self.Q3_1)

#=================================================Question 4=================================================
        self.load_image1_pushButton.clicked.connect(self.load_image1)
        self.load_image2_pushButton.clicked.connect(self.load_image2)
        self.keypoints_pushButton.clicked.connect(self.Q4_1)
        self.matched_keypoints_pushButton.clicked.connect(self.Q4_2)
#=================================================Question 5=================================================
        self.load_image_pushButton.clicked.connect(self.load_image)
        self.show_augmented_images_pushButton.clicked.connect(self.Q5_1)
        self.show_model_structure_pushButton.clicked.connect(self.Q5_2)
        self.show_acc_and_loss_pushButton.clicked.connect(self.Q5_3)
        self.inference_pushButton.clicked.connect(self.Q5_4)
        

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

    def show_on_label(self):
        q_image = QImage(self.image_inference.tobytes(), self.image_inference.width, self.image_inference.height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        pixmap = pixmap.scaled(label_width, label_height)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def load_image(self):
        self.inference_label.clear()
        image_path , _ = QFileDialog.getOpenFileName(None, "選擇圖片", "", "Images (*.png *.jpg *.bmp)")
        # self.image_inference = cv2_imread(image_path)
        self.image_inference = Image.open(image_path)
        print("\nLoad Image_inference")
        self.show_on_label()

    def Q1_1(self):
        title = "1-1"
        result = draw_corner()
        for img in result:
            cv2.namedWindow(title , cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title , 1500 , 1500)
            cv2.imshow(title , img)
            cv2.waitKey(500)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return

    def Q1_2(self):
        print("\nProcessing...")
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
        print("\nProcessing...")
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
        print("\nProcessing...")
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
        print("\nProcessing...")
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
                cv2.namedWindow("2-1", cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow("2-1", draw_image)
                cv2.waitKey(1000)
        cv2.destroyAllWindows()
        self.setEnabled(True)
    
    def Q2_2(self):
        print("\nProcessing...")
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
                cv2.namedWindow("2-2", cv2.WINDOW_GUI_EXPANDED)
                cv2.imshow("2-2", draw_image)
                cv2.waitKey(1000)
        cv2.destroyAllWindows() 
        self.setEnabled(True)

#=================================================Question 3=================================================
    def load_image_l(self):
        image_path , _ = QFileDialog.getOpenFileName(None, "選擇圖片", "", "Images (*.png *.jpg *.bmp)")
        self.image_L = cv2_imread(image_path)
        print("\nLoad Image_L")

    def load_image_r(self):
        image_path , _ = QFileDialog.getOpenFileName(None, "選擇圖片", "", "Images (*.png *.jpg *.bmp)")
        self.image_R = cv2_imread(image_path)
        print("\nLoad Image_L")
    
    def click_event(self,event, x, y, flags, params): 
        if(event == cv2.EVENT_LBUTTONDOWN):
            img = cv2.normalize(self.disparity_map, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            originalHeight, originalWidth = self.image_L.shape[:2]
            height, width = 400 , int((originalWidth / originalHeight) * 400)      
            x = int((x / width) * originalWidth)
            y = int((y / height) * originalHeight)
            if(img[y,x] == 0):
                print('Failure case')
                cv2.imshow('image_R',resizeImage(self.image_R))
                return
            dis = self.disparity_map[y,x] // 16
            x -= dis
            print('({},{}),dis:{:.2f}'.format(x,y,abs(dis)))
            img = cv2.circle(self.image_R.copy(), (x,y), 5, (0, 0, 255), -1)
            cv2.imshow('image_R', resizeImage(img))
            return
    
    def Q3_1(self):
        stereo_BM = cv2.StereoBM.create(numDisparities=256, blockSize=25)
        gray_image_L = cv2.cvtColor(self.image_L, cv2.COLOR_RGB2GRAY)
        gray_image_R = cv2.cvtColor(self.image_R, cv2.COLOR_RGB2GRAY)

        self.disparity_map = stereo_BM.compute(gray_image_L , gray_image_R)
        image = cv2.normalize(self.disparity_map, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        image =  image.astype(np.uint8)
        
        cv2.imshow('image_L',resizeImage(self.image_L))
        cv2.imshow('image_R',resizeImage(self.image_R))
        cv2.imshow('disparity',resizeImage(image))
        cv2.setMouseCallback('image_L', self.click_event) 
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        


#=================================================Question 4=================================================
    def load_image1(self):
        image_path , _ = QFileDialog.getOpenFileName(None, "選擇圖片", "", "Images (*.png *.jpg *.bmp)")
        self.image_1 = cv2_imread(image_path)
        print("\nLoad Image_1")

    def load_image2(self):
        image_path , _ = QFileDialog.getOpenFileName(None, "選擇圖片", "", "Images (*.png *.jpg *.bmp)")
        self.image_2 = cv2_imread(image_path)
        print("\nLoad Image_2")

    def Q4_1(self):
        print("\nProcessing...")

        sift = cv2.SIFT_create()
        gray_image = cv2.cvtColor(self.image_1, cv2.COLOR_RGB2GRAY)

        keypoints , descriptors = sift.detectAndCompute(gray_image,None)
        result = cv2.drawKeypoints(gray_image , keypoints , None , color=[0,255,0])
        cv2.namedWindow("4-1", cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("4-1", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def Q4_2(self):
        print("\nProcessing...")
        sift = cv2.SIFT_create()
        result = match_keypoints(self.image_1 , self.image_2)

        cv2.namedWindow("4-2", cv2.WINDOW_GUI_EXPANDED)
        cv2.imshow("4-2", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



#=================================================Question 5=================================================
    def load_model(self):
        model = VGG19BN().to(DEVICE)
        checkpoint_path = str(training_path.joinpath("model.pth"))
        model.load_state_dict(torch.load(checkpoint_path))
        # model = torch.load(model_path , map_location=DEVICE)
        model.eval()
        return model

    def preprocess(self , img):
        # img = Image.open(img_path)
        test_transforms = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
        image_tensor = test_transforms(img)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE) 
        return image_tensor

    def predict(self, model, test_data):
        output_tensor = model(test_data)
        probabilities = torch.nn.functional.softmax(output_tensor, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        predicted_label = label_dict[predicted_class]

        return predicted_label , probabilities

    def start_prediction(self , model):
        test_data = self.preprocess(self.image_inference)
        result , probabilities = self.predict(model , test_data)
        return result , probabilities
    
    
    def Q5_1(self):
        image_path = str( dataset_path.joinpath("Q5_Image").joinpath("Q5_1").joinpath("*.png") )
        all_path = glob.glob(image_path)
        images = []
        names = []
        augmented_images = []

        for path in all_path:
            filename = str(Path(path).stem)
            # print(filename)
            img = Image.open(path)
            images.append(img)
            names.append(filename)

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
            transforms.RandomVerticalFlip(),  # 随机水平翻转
            transforms.RandomRotation(30) # 随机旋转（-30到30度之间）
        ])

        for image in images:
            augmented_image = transform(image)
            augmented_images.append(augmented_image)
        
        fig = plt.figure("5-1",figsize=(15,15))
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.title(names[i]) 
            plt.imshow(augmented_images[i])

        img = plot_to_image(fig)
        plt.close(fig)
        cv2.imshow("5-1",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Q5_2(self):
        model = VGG19BN().to(DEVICE)
        summary(model, input_size=(3, 32 , 32) )

    def Q5_3(self):
        figure_path = str(training_path.joinpath("accuracy and loss.png"))
        figure = cv2_imread(figure_path)
        cv2.imshow("5-3",figure)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def Q5_4(self):
        print("\nStart Inference")
        model = self.load_model()
        result , probabilities = self.start_prediction(model)
        text = "Predicted : " + result
        self.inference_label.setText(text)
        
        # 绘制概率分布图
        classes = list(label_dict.values())
        probs = probabilities.cpu().detach().numpy()
        
        fig = plt.figure("5-4", figsize=(15, 15))
        plt.bar(classes, probs, color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.title('Class Probability Distribution')

        # 将fig转换为图像
        image = plot_to_image(fig)

        # 使用OpenCV显示图像
        cv2.imshow('Probability Distribution', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec())