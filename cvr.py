# Cardiovasculair Risicomanagement
# https://www.nhg.org/standaarden/volledig/cardiovasculair-risicomanagement#idp1219088
# Inputs:
#   Geslacht
#   Leeftijd
#   Roken of niet roken
#   SBD
#   Ratio totaal cholesterol
# Output:
#   Kans (0-1)
import tensorflow as tf
import data as d
import numpy as np

x = tf.placeholder("float", shape=[None, 5])
y_ = tf.placeholder("float", shape=[None, 1])

W = tf.Variable(tf.random_uniform([5, 1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = tf.matmul(x, W) + b

loss = tf.reduce_mean(tf.square(y - y_))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

trainData = d.Data()

for i in xrange(2001):
    session.run(train, feed_dict={x: trainData.x_data, y_: trainData.y_data})
    if i % 50 == 0:
        print i, session.run(W), session.run(b)

print "\n"
i = 25
x_test = np.array([trainData.x_data[i]], dtype=float)
y_test = np.array([trainData.y_data[i]], dtype=float)
print session.run(y, feed_dict={x: x_test})
print y_test
