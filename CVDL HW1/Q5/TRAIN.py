import silence_tensorflow.auto
import tensorflow as tf
import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping

(train_data, train_label), (test_data, test_label) = keras.datasets.cifar10.load_data()
image = []
label = []
BATCH_SIZE = 128
LEARNING_RATE = 0.00001
OPTIMIZER = "Adam"

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

def train(model):
    history = model.fit(train_data , train_label , batch_size=BATCH_SIZE , epochs=30 , shuffle=True , validation_split=0.1)
    save_path = os.path.join(os.path.dirname(__file__),"model.h5")
    model.save(save_path)
    #accuracy
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend(["train", "validation"], loc="lower right")
    plt.show()
    # loss graph
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train", "validation"], loc="upper right")
    plt.show()
    return

if __name__ == "__main__": 
    model = build_model()
    train(model)