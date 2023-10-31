# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'HW1ypdhhr.ui'
##
## Created by: Qt User Interface Compiler version 6.6.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QGraphicsView, QGroupBox,
    QHBoxLayout, QLineEdit, QMainWindow, QMenuBar,
    QPushButton, QSizePolicy, QStatusBar, QVBoxLayout,
    QWidget)

class UI(object):
    def setup_UI(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(912, 599)
        MainWindow.setMinimumSize(QSize(200, 250))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName(u"widget")
        self.widget.setGeometry(QRect(0, 0, 901, 571))
        self.verticalLayoutWidget_8 = QWidget(self.widget)
        self.verticalLayoutWidget_8.setObjectName(u"verticalLayoutWidget_8")
        self.verticalLayoutWidget_8.setGeometry(QRect(0, 0, 911, 571))
        self.verticalLayout_2 = QVBoxLayout(self.verticalLayoutWidget_8)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.groupBox = QGroupBox(self.verticalLayoutWidget_8)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setMinimumSize(QSize(161, 211))
        self.groupBox.setMaximumSize(QSize(200, 250))
        self.groupBox.setAlignment(Qt.AlignCenter)
        self.verticalLayoutWidget = QWidget(self.groupBox)
        self.verticalLayoutWidget.setObjectName(u"verticalLayoutWidget")
        self.verticalLayoutWidget.setGeometry(QRect(40, 20, 131, 231))
        self.verticalLayout = QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setSpacing(5)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.load_foler_pushButton = QPushButton(self.verticalLayoutWidget)
        self.load_foler_pushButton.setObjectName(u"load_foler_pushButton")

        self.verticalLayout.addWidget(self.load_foler_pushButton)

        self.load_image_l_pushButton = QPushButton(self.verticalLayoutWidget)
        self.load_image_l_pushButton.setObjectName(u"load_image_l_pushButton")

        self.verticalLayout.addWidget(self.load_image_l_pushButton)

        self.load_image_r_pushButton = QPushButton(self.verticalLayoutWidget)
        self.load_image_r_pushButton.setObjectName(u"load_image_r_pushButton")

        self.verticalLayout.addWidget(self.load_image_r_pushButton)


        self.horizontalLayout_2.addWidget(self.groupBox)

        self.groupBox_2 = QGroupBox(self.verticalLayoutWidget_8)
        self.groupBox_2.setObjectName(u"groupBox_2")
        self.groupBox_2.setMinimumSize(QSize(200, 250))
        self.groupBox_2.setMaximumSize(QSize(200, 250))
        self.groupBox_2.setAlignment(Qt.AlignCenter)
        self.verticalLayoutWidget_2 = QWidget(self.groupBox_2)
        self.verticalLayoutWidget_2.setObjectName(u"verticalLayoutWidget_2")
        self.verticalLayoutWidget_2.setGeometry(QRect(30, 20, 151, 231))
        self.verticalLayout_7 = QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.find_corners_pushButton = QPushButton(self.verticalLayoutWidget_2)
        self.find_corners_pushButton.setObjectName(u"find_corners_pushButton")

        self.verticalLayout_7.addWidget(self.find_corners_pushButton)

        self.find_intrinsic_pushButton = QPushButton(self.verticalLayoutWidget_2)
        self.find_intrinsic_pushButton.setObjectName(u"find_intrinsic_pushButton")

        self.verticalLayout_7.addWidget(self.find_intrinsic_pushButton)

        self.groupBox_3 = QGroupBox(self.verticalLayoutWidget_2)
        self.groupBox_3.setObjectName(u"groupBox_3")
        self.groupBox_3.setAlignment(Qt.AlignCenter)
        self.verticalLayoutWidget_7 = QWidget(self.groupBox_3)
        self.verticalLayoutWidget_7.setObjectName(u"verticalLayoutWidget_7")
        self.verticalLayoutWidget_7.setGeometry(QRect(10, 20, 131, 81))
        self.verticalLayout_8 = QVBoxLayout(self.verticalLayoutWidget_7)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.comboBox = QComboBox(self.verticalLayoutWidget_7)
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setMaxVisibleItems(15)

        self.verticalLayout_8.addWidget(self.comboBox)

        self.find_extrinsic_pushButton = QPushButton(self.verticalLayoutWidget_7)
        self.find_extrinsic_pushButton.setObjectName(u"find_extrinsic_pushButton")

        self.verticalLayout_8.addWidget(self.find_extrinsic_pushButton)


        self.verticalLayout_7.addWidget(self.groupBox_3)

        self.find_distortion_pushButton = QPushButton(self.verticalLayoutWidget_2)
        self.find_distortion_pushButton.setObjectName(u"find_distortion_pushButton")

        self.verticalLayout_7.addWidget(self.find_distortion_pushButton)

        self.show_undistorted_result_pushButton = QPushButton(self.verticalLayoutWidget_2)
        self.show_undistorted_result_pushButton.setObjectName(u"show_undistorted_result_pushButton")

        self.verticalLayout_7.addWidget(self.show_undistorted_result_pushButton)


        self.horizontalLayout_2.addWidget(self.groupBox_2)

        self.groupBox_6 = QGroupBox(self.verticalLayoutWidget_8)
        self.groupBox_6.setObjectName(u"groupBox_6")
        self.groupBox_6.setMinimumSize(QSize(161, 211))
        self.groupBox_6.setMaximumSize(QSize(200, 250))
        self.groupBox_6.setAlignment(Qt.AlignCenter)
        self.show_words_vertical_pushButton = QPushButton(self.groupBox_6)
        self.show_words_vertical_pushButton.setObjectName(u"show_words_vertical_pushButton")
        self.show_words_vertical_pushButton.setGeometry(QRect(41, 186, 151, 24))
        self.lineEdit = QLineEdit(self.groupBox_6)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QRect(41, 58, 141, 20))
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit.sizePolicy().hasHeightForWidth())
        self.lineEdit.setSizePolicy(sizePolicy)
        self.show_words_on_board_pushButton = QPushButton(self.groupBox_6)
        self.show_words_on_board_pushButton.setObjectName(u"show_words_on_board_pushButton")
        self.show_words_on_board_pushButton.setGeometry(QRect(41, 120, 149, 24))

        self.horizontalLayout_2.addWidget(self.groupBox_6)

        self.groupBox_5 = QGroupBox(self.verticalLayoutWidget_8)
        self.groupBox_5.setObjectName(u"groupBox_5")
        self.groupBox_5.setMinimumSize(QSize(161, 211))
        self.groupBox_5.setMaximumSize(QSize(200, 250))
        self.groupBox_5.setAlignment(Qt.AlignCenter)
        self.verticalLayoutWidget_4 = QWidget(self.groupBox_5)
        self.verticalLayoutWidget_4.setObjectName(u"verticalLayoutWidget_4")
        self.verticalLayoutWidget_4.setGeometry(QRect(24, 20, 171, 231))
        self.verticalLayout_3 = QVBoxLayout(self.verticalLayoutWidget_4)
        self.verticalLayout_3.setSpacing(5)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.stereo_disparity_map_pushButton = QPushButton(self.verticalLayoutWidget_4)
        self.stereo_disparity_map_pushButton.setObjectName(u"stereo_disparity_map_pushButton")

        self.verticalLayout_3.addWidget(self.stereo_disparity_map_pushButton)


        self.horizontalLayout_2.addWidget(self.groupBox_5)


        self.verticalLayout_2.addLayout(self.horizontalLayout_2)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.groupBox_7 = QGroupBox(self.verticalLayoutWidget_8)
        self.groupBox_7.setObjectName(u"groupBox_7")
        self.groupBox_7.setMinimumSize(QSize(161, 211))
        self.groupBox_7.setMaximumSize(QSize(200, 250))
        self.groupBox_7.setAlignment(Qt.AlignCenter)
        self.verticalLayoutWidget_6 = QWidget(self.groupBox_7)
        self.verticalLayoutWidget_6.setObjectName(u"verticalLayoutWidget_6")
        self.verticalLayoutWidget_6.setGeometry(QRect(40, 20, 140, 231))
        self.verticalLayout_6 = QVBoxLayout(self.verticalLayoutWidget_6)
        self.verticalLayout_6.setSpacing(5)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.load_image1_pushButton = QPushButton(self.verticalLayoutWidget_6)
        self.load_image1_pushButton.setObjectName(u"load_image1_pushButton")

        self.verticalLayout_6.addWidget(self.load_image1_pushButton)

        self.load_image2_pushButton = QPushButton(self.verticalLayoutWidget_6)
        self.load_image2_pushButton.setObjectName(u"load_image2_pushButton")

        self.verticalLayout_6.addWidget(self.load_image2_pushButton)

        self.keypoints_pushButton = QPushButton(self.verticalLayoutWidget_6)
        self.keypoints_pushButton.setObjectName(u"keypoints_pushButton")

        self.verticalLayout_6.addWidget(self.keypoints_pushButton)

        self.matched_keypoints_pushButton = QPushButton(self.verticalLayoutWidget_6)
        self.matched_keypoints_pushButton.setObjectName(u"matched_keypoints_pushButton")

        self.verticalLayout_6.addWidget(self.matched_keypoints_pushButton)


        self.horizontalLayout_3.addWidget(self.groupBox_7)

        self.groupBox_8 = QGroupBox(self.verticalLayoutWidget_8)
        self.groupBox_8.setObjectName(u"groupBox_8")
        self.groupBox_8.setMinimumSize(QSize(0, 0))
        self.groupBox_8.setMaximumSize(QSize(800, 600))
        self.groupBox_8.setAlignment(Qt.AlignCenter)
        self.verticalLayoutWidget_3 = QWidget(self.groupBox_8)
        self.verticalLayoutWidget_3.setObjectName(u"verticalLayoutWidget_3")
        self.verticalLayoutWidget_3.setGeometry(QRect(120, 30, 174, 221))
        self.verticalLayout_11 = QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.verticalLayout_11.setContentsMargins(0, 0, 0, 0)
        self.load_image_pushButton = QPushButton(self.verticalLayoutWidget_3)
        self.load_image_pushButton.setObjectName(u"load_image_pushButton")

        self.verticalLayout_11.addWidget(self.load_image_pushButton)

        self.show_augmented_images_pushButton = QPushButton(self.verticalLayoutWidget_3)
        self.show_augmented_images_pushButton.setObjectName(u"show_augmented_images_pushButton")

        self.verticalLayout_11.addWidget(self.show_augmented_images_pushButton)

        self.show_model_structure_pushButton = QPushButton(self.verticalLayoutWidget_3)
        self.show_model_structure_pushButton.setObjectName(u"show_model_structure_pushButton")

        self.verticalLayout_11.addWidget(self.show_model_structure_pushButton)

        self.show_acc_and_loss_pushButton = QPushButton(self.verticalLayoutWidget_3)
        self.show_acc_and_loss_pushButton.setObjectName(u"show_acc_and_loss_pushButton")

        self.verticalLayout_11.addWidget(self.show_acc_and_loss_pushButton)

        self.inference_pushButton = QPushButton(self.groupBox_8)
        self.inference_pushButton.setObjectName(u"inference_pushButton")
        self.inference_pushButton.setGeometry(QRect(420, 20, 200, 25))
        self.graphicsView = QGraphicsView(self.groupBox_8)
        self.graphicsView.setObjectName(u"graphicsView")
        self.graphicsView.setGeometry(QRect(420, 50, 200, 200))

        self.horizontalLayout_3.addWidget(self.groupBox_8)


        self.verticalLayout_2.addLayout(self.horizontalLayout_3)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 912, 21))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.groupBox.setTitle(QCoreApplication.translate("MainWindow", u"Load Image", None))
        self.load_foler_pushButton.setText(QCoreApplication.translate("MainWindow", u"Load folder", None))
        self.load_image_l_pushButton.setText(QCoreApplication.translate("MainWindow", u"Load Image_L", None))
        self.load_image_r_pushButton.setText(QCoreApplication.translate("MainWindow", u"Load Image_R", None))
        self.groupBox_2.setTitle(QCoreApplication.translate("MainWindow", u"1. Classification", None))
        self.find_corners_pushButton.setText(QCoreApplication.translate("MainWindow", u"1.1 Find corners", None))
        self.find_intrinsic_pushButton.setText(QCoreApplication.translate("MainWindow", u"1.2 Find intrinsic", None))
        self.groupBox_3.setTitle(QCoreApplication.translate("MainWindow", u"1.3 Find extrinsic", None))
        self.find_extrinsic_pushButton.setText(QCoreApplication.translate("MainWindow", u"Find extrinsic", None))
        self.find_distortion_pushButton.setText(QCoreApplication.translate("MainWindow", u"1.4 Find distortion", None))
        self.show_undistorted_result_pushButton.setText(QCoreApplication.translate("MainWindow", u"1.5 Show result", None))
        self.groupBox_6.setTitle(QCoreApplication.translate("MainWindow", u"2. Augmented reality", None))
        self.show_words_vertical_pushButton.setText(QCoreApplication.translate("MainWindow", u"2.2 show words vertical", None))
        self.show_words_on_board_pushButton.setText(QCoreApplication.translate("MainWindow", u"2.1 show words on board", None))
        self.groupBox_5.setTitle(QCoreApplication.translate("MainWindow", u"3. Stereo disparity map", None))
        self.stereo_disparity_map_pushButton.setText(QCoreApplication.translate("MainWindow", u"3.1 stereo disparity map", None))
        self.groupBox_7.setTitle(QCoreApplication.translate("MainWindow", u"4. SIFT", None))
        self.load_image1_pushButton.setText(QCoreApplication.translate("MainWindow", u"Load Image1", None))
        self.load_image2_pushButton.setText(QCoreApplication.translate("MainWindow", u"Load Image2", None))
        self.keypoints_pushButton.setText(QCoreApplication.translate("MainWindow", u"4.1 Keypoints", None))
        self.matched_keypoints_pushButton.setText(QCoreApplication.translate("MainWindow", u"4.2 Matched keypoints", None))
        self.groupBox_8.setTitle(QCoreApplication.translate("MainWindow", u"3. Stereo disparity map", None))
        self.load_image_pushButton.setText(QCoreApplication.translate("MainWindow", u"Load Image", None))
        self.show_augmented_images_pushButton.setText(QCoreApplication.translate("MainWindow", u"5.1 Show augmented images", None))
        self.show_model_structure_pushButton.setText(QCoreApplication.translate("MainWindow", u"5.2 Show model structure", None))
        self.show_acc_and_loss_pushButton.setText(QCoreApplication.translate("MainWindow", u"5.3 Show acc and loss", None))
        self.inference_pushButton.setText(QCoreApplication.translate("MainWindow", u"5.4 Inference", None))
    # retranslateUi

