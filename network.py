import argparse
import os
import shutil
import time
import random
import numpy as np
import math
import sys
import datetime
import h5py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Flatten, Dropout, Dense, Input, Conv2D, Reshape
from tensorflow.keras.optimizers import Adam, Nadam

from utils import dotdict

## Much credit to the alpha-zero-general library and examples. 
class Network():
    """
    This class contains all the functions and models and things related to the neural netork.
    """
    def __init__(self):
        # game params

        self.lr = 0.001
        self.dropout = 0.3
        self.epochs = 10
        self.batch_size = 128
        self.num_channels = 512
        self.action_space = 8192

        # Neural Net
        self.input_boards = Input(shape=(8, 8, 12))    # s: batch_size x board_x x number of possible pieces

        x_image = Reshape((8, 8, 12, 1))(self.input_boards)
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same', kernel_initializer='random_normal')(x_image)))
        h_conv2 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same', kernel_initializer='random_normal')(h_conv1))) 
        h_conv3 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='valid', kernel_initializer='random_normal')(h_conv2)))
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='valid', kernel_initializer='random_normal')(h_conv3)))
        h_conv5_flat = Flatten()(h_conv4)
        s_fc1 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv5_flat))))
        s_fc2 = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))
        self.pi = Dense(self.action_space, activation='softmax', name='pi')(s_fc2)
        self.v = Dense(1, activation='tanh', name='v')(s_fc2)

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(self.lr), metrics=['accuracy'])
        print(self.model.summary())

    def train(self, examples):
            """
            examples: list of examples, each example is of form (board, pi, v)
            """
            input_boards, target_pis, target_vs = list(zip(*examples))
            input_boards = np.asarray(input_boards)
            target_pis = np.asarray(target_pis)
            target_vs = np.asarray(target_vs)
            self.model.fit(x = input_boards, y = [target_pis, target_vs], batch_size = self.batch_size, epochs = self.epochs)

    def predict(self, board):
        """
        board: np array with board
        """
        # timing
        start = time.time()

        # preparing input
        board = board[np.newaxis, :, :]

        # run
        pi, v = self.model.predict(board)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.h5'):
        filepath = os.path.join(folder, filename)
        self.model.save(filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.h5'):
        filepath = os.path.join(folder, filename)
        self.model.load_weights(filepath)