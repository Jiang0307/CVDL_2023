import silence_tensorflow.auto
import sys
import numpy as np
import cv2
import tensorflow as tf
import keras
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import torchvision.models as models
from torchsummary import summary
from keras.applications import VGG16
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping
from math import sqrt
from PyQt5 import QtWidgets, QtGui, QtCore
from UI_Q5 import UI

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

def build_model():
    backbone = VGG16(weights="imagenet" , include_top=False , input_shape=(32,32,3))
    model = Sequential()
    for layer in backbone.layers:
        model.add(layer)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation="relu", name="full_connected_1"))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu", name="full_connected_2"))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation="softmax", name="output"))
    model.compile(optimizer=OPTIMIZER , loss="sparse_categorical_crossentropy" , metrics=["accuracy"])
    return model

(train_data, train_label), (test_data, test_label) = keras.datasets.cifar10.load_data()
BATCH_SIZE = 128
LEARNING_RATE = 0.00001
OPTIMIZER = "Adam"
image = []
label = []
model = build_model()

class MainWindow(QtWidgets.QMainWindow,UI):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.UI_SET(self)
        self.button_1.clicked.connect(self.HW1_5_1)
        self.button_2.clicked.connect(self.HW1_5_2)
        self.button_3.clicked.connect(self.HW1_5_3)
        self.button_4.clicked.connect(self.HW1_5_4)
        self.button_5.clicked.connect(self.HW1_5_5)
    
    def HW1_5_1(self):
        class_name = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]    
        for i in range(9):
            index = random.randint(0,train_data.shape[0])
            image.append(train_data[index])
            label.append(train_label[index])

        fig = plt.figure("5-1",figsize=(15,15))
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.title(class_name[label[i][0]]) 
            plt.imshow(image[i])

        img = plot_to_image(fig)
        plt.close(fig)
        cv2.imshow("5-1",img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def HW1_5_2(self):
        print("\nhyperparameters : ")
        print("batch size\t: ",BATCH_SIZE)
        print("learning rate\t: ",LEARNING_RATE)
        print("optimizer\t: ",OPTIMIZER,"\n")
        return

    def HW1_5_3(self):
        model.summary()
        return

    def HW1_5_4(self):
        path_1 = os.path.join(os.path.dirname(__file__),"DATA","accuracy.png")
        accuracy = cv2_imread(path_1)

        path_2 = os.path.join(os.path.dirname(__file__),"DATA","loss.png")
        loss = cv2_imread(path_2)

        result = np.vstack((accuracy,loss))
        cv2.imshow("5-4",result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    def HW1_5_5(self):
        path = os.path.join(os.path.dirname(__file__),r"DATA",r"model.h5")
        model = keras.models.load_model(path)
        index = int(self.textEdit.text())
        img = test_data[index]
        test_img = np.expand_dims(img,axis=0)

        class_name = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        ans = model.predict(test_img)

        fig = plt.figure("5-5",figsize=(10,10))
        plt.subplot(1, 1, 1)
        plt.bar(class_name,ans[0])
        plt.xlabel("class name")
        plt.ylabel("probability")
        plt.xticks(fontsize=10)

        fig_4_channel = plot_to_image(fig)
        fig_3_channel = cv2.cvtColor(fig_4_channel, cv2.COLOR_RGBA2RGB)
        img = cv2.resize(img, (fig_3_channel.shape[0],fig_3_channel.shape[0]))
        result = np.hstack((img,fig_3_channel))
        plt.close()
        cv2.imshow("5-5",result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())