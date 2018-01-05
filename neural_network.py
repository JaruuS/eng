import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from data_processing import get_next_batch



train_dataset = numpy.loadtxt(open('D:\\eng\\data_preprocessed_python\\train_dataset.txt', 'rb'))
train_labels = numpy.loadtxt(open('D:\\eng\\data_preprocessed_python\\train_labels.txt', 'rb'))
test_dataset = numpy.loadtxt(open('D:\\eng\\data_preprocessed_python\\test_dataset.txt', 'rb'))
test_labels = numpy.loadtxt(open('D:\\eng\\data_preprocessed_python\\test_labels.txt', 'rb'))
batch_size = 64
steps = 8096
test_rate = 512
hidden_layer_1 = 5
test_acc = list()

init = tf.global_variables_initializer()
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 504])
Y_ = tf.placeholder(tf.float32, [None, 12])

W1 = tf.Variable(tf.truncated_normal([504, hidden_layer_1], stddev=0.1))
B1 = tf.Variable(tf.zeros([hidden_layer_1]))
W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, 12], stddev=0.1))
B2 = tf.Variable(tf.zeros([12]))

Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y1d = tf.nn.dropout(Y1, pkeep)
Y2 = tf.nn.relu(tf.matmul(Y1d, W2) + B2)
Y2d = tf.nn.dropout(Y2, pkeep)

Ylogits = tf.matmul(Y1d, W2) + B2
Y = tf.nn.softmax(Ylogits)

cost = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cost = tf.reduce_mean(cost) * 100

optimizer = tf.train.AdamOptimizer(1.0)
train_step = optimizer.minimize(cost)

correct_prediction = tf.equal(tf.arg_max(Y, 1), tf.arg_max(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

test_data = {X: test_dataset, Y_: test_labels, pkeep: [0.1]}
for i in range(steps):
    batch_X = get_next_batch(i,batch_size,train_dataset)
    batch_Y = get_next_batch(i,batch_size,train_labels)

    train_data = {X: batch_X, Y_: batch_Y, pkeep: [0.1], lr: 0.1}

    if i % test_rate == 0:
        a = sess.run(accuracy, feed_dict=test_data)
        test_acc.append(a)

    sess.run(train_step, feed_dict=train_data)


a = sess.run(accuracy, feed_dict=test_data)
test_acc.append(a)
plt.plot(test_acc)
plt.title('Końcowa skuteczność sieci: {0}'.format(a))
plt.show()



