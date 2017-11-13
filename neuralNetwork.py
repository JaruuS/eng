import numpy as np
import tensorflow as tf
import time
import math

from openFile import split_dataset_into_training_and_test

X = tf.placeholder(tf.float32, [None, 504])

init = tf.global_variables_initializer()

Y_ = tf.placeholder(tf.float32, [None, 12])

lr = tf.placeholder(tf.float32)

pkeep = tf.placeholder(tf.float32)

L = 5
M = 100
N = 50
O = 25

W1 = tf.Variable(tf.truncated_normal([504, L], stddev=0.1))
B1 = tf.Variable(tf.zeros([L]))
W2 = tf.Variable(tf.truncated_normal([L, 12], stddev=0.1))
B2 = tf.Variable(tf.zeros([12]))
#W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
#B3 = tf.Variable(tf.zeros([N]))
#W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
#B4 = tf.Variable(tf.zeros([O]))
#W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))
#B5 = tf.Variable(tf.zeros([10]))

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)
#Y3 = tf.nn.relu(tf.matmul(Y2d, W3) + B3)
#Y3d = tf.nn.dropout(Y3, pkeep)
#Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
#Y4d = tf.nn.dropout(Y4, pkeep)
Ylogits = tf.matmul(Y1d, W2) + B2
Y = tf.nn.softmax(Ylogits)

cost = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cost = tf.reduce_mean(cost) * 100

optimizer = tf.train.AdamOptimizer(0.001)
train_step = optimizer.minimize(cost)

correct_prediction = tf.equal(tf.arg_max(Y, 1), tf.arg_max(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
labels = np.loadtxt(open('D:\eng\data_preprocessed_python\hot_vectors.txt', 'rb'))
dataset = np.loadtxt(open('D:\eng\data_preprocessed_python\dataset.txt', 'rb'))
data = split_dataset_into_training_and_test(dataset, labels)

for i in range(32):
    # load batch of images and correct answers
    batch_X = data[0]
    batch_Y = data[1]

    train_data = {X: batch_X[:32, :], Y_: batch_Y[:32, :], pkeep: [0.75], lr: 0.001}
    for j in range(0, 32):
        batch_X = np.delete(batch_X,0,0)
        batch_Y = np.delete(batch_Y, 0, 0)

    # train
    sess.run(train_step, feed_dict=train_data)

a, c = sess.run([accuracy, cost], feed_dict=train_data)

test_data = {X: data[2], Y_: data[3], pkeep: [1]}
a, c = sess.run([accuracy, cost], feed_dict=test_data)
print('Accuracy {0}'.format(a))
