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

from tensorflow import Tensor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Flatten, Dropout, Dense, Input, Conv2D, Reshape, Add, ReLU
from tensorflow.keras.optimizers import Adam, Nadam

from utils import dotdict
from network import Network

def conv_block(feat_maps_out, prev):
    prev = BatchNormalization(axis=1)(prev)
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, 3, strides=1, padding='same')(prev) 
    prev = BatchNormalization(axis=1)(prev)
    prev = Activation('relu')(prev)
    prev = Conv2D(feat_maps_out, 3, strides=1, padding='same')(prev) 
    return prev

def skip_block(feat_maps_in, feat_maps_out, prev):
    if feat_maps_in != feat_maps_out:
        # This adds in a 1x1 convolution on shortcuts that map between an uneven amount of channels
        prev = Conv2D(feat_maps_out, 1, 1, border_mode='same')(prev)
    return prev

def residual(feat_maps_in, feat_maps_out, prev_layer):
    skip = skip_block(feat_maps_in, feat_maps_out, prev_layer)
    conv = conv_block(feat_maps_out, prev_layer)
    return Add()([skip, conv]) # the residual connection

## Much credit to the alpha-zero-general library and examples. 
class NetworkV2(Network):
    """
    This class contains all the functions and models and things related to the neural netork.
    """

    def __init__(self):
        # game params

        self.lr = 0.001
        self.dropout = 0.3
        self.epochs = 20
        self.batch_size = 128
        self.num_channels = 256
        self.action_space = 8192

        # Neural Net
        self.input_boards = Input(shape=(8, 8, 12))    # s: batch_size x board_x x number of possible pieces

        x_image = Reshape((8, 8, 12, 1))(self.input_boards)
        # Convolutional block
        t = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.num_channels, 3, padding='same', kernel_initializer='random_normal')(x_image)))
        # Create residual blocks
        for _ in range(5):
            t = residual(self.num_channels, self.num_channels, t)

        flat = Flatten()(t)
        head = Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(256)(flat))))
        # policy_head = (Dropout(self.dropout)(Activation('relu')(BatchNormalization(axis=3)(Conv2D(2, 1, 1, padding='same')(t)))))
        # value_head = (Dropout(self.dropout)((Activation('relu')(Dense(256)(t)))))

        self.pi = Dense(self.action_space, activation='softmax', name='pi')(head)
        self.v = Dense(1, activation='tanh', name='v')(head)

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