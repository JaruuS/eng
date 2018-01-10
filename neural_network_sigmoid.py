import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

def get_next_batch(i, batch, data):
    return data[0 + batch * (i % (1024 // batch)):0 + batch * (i % (1024 // batch)) + batch]

train_dataset = numpy.loadtxt(open('/home/jaruus/Pulpit/train_dataset.txt', 'rb'))
train_labels = numpy.loadtxt(open('/home/jaruus/Pulpit/train_labels.txt', 'rb'))
test_dataset = numpy.loadtxt(open('/home/jaruus/Pulpit/test_dataset.txt', 'rb'))
test_labels = numpy.loadtxt(open('/home/jaruus/Pulpit/test_labels.txt', 'rb'))
batch_size = 64
steps = 1024
test_rate = 32
hidden_layer_1 = 14
test_acc = list()
test_cost = list()

init = tf.global_variables_initializer()
lr = tf.placeholder(tf.float32)
#pkeep = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 2])
Y_ = tf.placeholder(tf.float32, [None, 12])

W1 = tf.Variable(tf.truncated_normal([2, hidden_layer_1], stddev=0.5))
B1 = tf.Variable(tf.zeros([hidden_layer_1]))
W2 = tf.Variable(tf.truncated_normal([hidden_layer_1, 12], stddev=0.5))
B2 = tf.Variable(tf.zeros([12]))

Y1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)

Ylogits = tf.matmul(Y1, W2) + B2
Y = tf.nn.sigmoid(Ylogits)

cost = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cost = tf.reduce_mean(cost) * 10

optimizer = tf.train.AdamOptimizer(1)
train_step = optimizer.minimize(cost)

correct_prediction = tf.equal(tf.arg_max(Y, 1), tf.arg_max(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

test_data = {X: test_dataset, Y_: test_labels}
for i in range(steps):
    batch_X = get_next_batch(i,batch_size,train_dataset)
    batch_Y = get_next_batch(i,batch_size,train_labels)

    train_data = {X: batch_X, Y_: batch_Y}

    #if i % test_rate == 0:
    b, a = sess.run([cost,accuracy], feed_dict=test_data)
    test_acc.append(a)
    test_cost.append(b)
	

    sess.run(train_step, feed_dict=train_data)


a = sess.run(accuracy, feed_dict=test_data)
test_acc.append(a)

print(a)
plt.figure(1)
plt.subplot(211)
plt.title('Wykres zmiany skutecznoscci w czasie trenowania')
plt.plot(test_acc)

plt.subplot(212)
plt.title('Wykres zmiany funkcji kosztu w czasie trenowania')
plt.plot(test_cost)
plt.show()



