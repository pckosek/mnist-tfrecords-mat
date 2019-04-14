# ----------------------------------------------
# setup.py
# 
# read mnist data from MATLAB, and convert
#  into tfrecords objects
#
# Paul Kosek


import numpy as np
import tensorflow as tf
from easy_tfrecords import create_tfrecords, easy_tfrecords as records
import os
import scipy.io as sio


# -----------------------
# TEST DATA

# FETCH DATA FROM MATLAB
train = sio.loadmat('..\\data\\mat\\train.mat')

train_x0  = np.array( train['train_x0'], dtype=np.float32 )
train_y0  = np.array( train['train_y0'], dtype=np.float32 )

train_x1  = np.array( train['train_x1'], dtype=np.float32 )
train_y1  = np.array( train['train_y1'], dtype=np.float32 )

train_x2  = np.array( train['train_x2'], dtype=np.float32 )
train_y2  = np.array( train['train_y2'], dtype=np.float32 )

train_x3  = np.array( train['train_x3'], dtype=np.float32 )
train_y3  = np.array( train['train_y3'], dtype=np.float32 )

train_x4  = np.array( train['train_x4'], dtype=np.float32 )
train_y4  = np.array( train['train_y4'], dtype=np.float32 )

train_x5  = np.array( train['train_x5'], dtype=np.float32 )
train_y5  = np.array( train['train_y5'], dtype=np.float32 )

# # CREATE AND SAVE TO SOME TFRECORDS FILES
create_tfrecords('..\\data\\tf\\mnist_train_0_0.tf', x=train_x0, y=train_y0)
create_tfrecords('..\\data\\tf\\mnist_train_1_5.tf', x=train_x1, y=train_y1)
create_tfrecords('..\\data\\tf\\mnist_train_2_5.tf', x=train_x2, y=train_y2)
create_tfrecords('..\\data\\tf\\mnist_train_3_5.tf', x=train_x3, y=train_y3)
create_tfrecords('..\\data\\tf\\mnist_train_4_5.tf', x=train_x4, y=train_y4)
create_tfrecords('..\\data\\tf\\mnist_train_5_5.tf', x=train_x5, y=train_y5)


# -----------------------
# TEST DATA

test = sio.loadmat('..\\data\\mat\\test.mat')

matlab_x  = np.array( test['test_x'], dtype=np.float32 )
matlab_y  = np.array( test['test_y'], dtype=np.float32 )

# CREATE AND SAVE TO A TFRECORDS FILE
create_tfrecords('..\\data\\tf\\mnist_test_0_0.tf', x=matlab_x, y=matlab_y)