import numpy as np
import tensorflow as tf
from easy_tfrecords import create_tfrecords, easy_tfrecords as records
import os

# disable tensorflow logging stuff
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

BATCH_SIZE = 1


# INSTANTIATE THE RECORDS OBJECT
rec = records(files=['mnist_test_0_0.tf'],
  shuffle=False,
  batch_size=BATCH_SIZE, 
  keys=['x', 'y'])

next_factory = rec.get_next_factory()

batch_x = next_factory['x']
batch_y = next_factory['y']


with tf.Session() as sess:

  sess.run(rec.get_initializer())

  for n in range(1):
    print('-----------------------------------')
    print('n => {}\n'.format(n))

    x_eval, y_eval = sess.run( [batch_x, batch_y] )
    print('x_eval => {}'.format(x_eval))
    print('y_eval => {}'.format(y_eval))

sess.close()


