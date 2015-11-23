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
from __future__ import print_function
import tensorflow as tf
import data

x = tf.placeholder("float", shape=[None, 5])
y_ = tf.placeholder("float", shape=[None, 1])

hiddenLayerSize = 16
W1 = tf.Variable(tf.random_uniform([5, hiddenLayerSize]))
W2 = tf.Variable(tf.random_uniform([hiddenLayerSize, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.sigmoid(tf.matmul(tf.sigmoid(tf.matmul(x, W1)), W2))

loss = tf.reduce_mean(tf.square(y - y_))
train = tf.train.GradientDescentOptimizer(0.8).minimize(loss)

init = tf.initialize_all_variables()
session = tf.Session()
session.run(init)

# Train data
d = data.Data()


# Define train function
def train_network():
    for i in xrange(20001):
        session.run(train, feed_dict={x: d.x_data, y_: d.y_data})
        if i % 500 == 0:
            y_prediction = session.run(y, feed_dict={x: d.x_data})
            print(i, "\t", session.run(loss, feed_dict={y: y_prediction, y_: d.y_data}))


# Define test function
def test():
    for i in range(d.x_data.shape[0]):
        x_test = [d.x_data[i]]
        y_test = [d.y_data[i]]
        print(session.run(y, feed_dict={x: x_test})[0][0], "\t", y_test[0][0])


# Print in table-form
def test_table():
    yHat = session.run(y, feed_dict={x: d.x_data, y_: d.y_data})
    yHat *= 50
    for i in range(24):
        for j in range(4):
            for k in range(5):
                string = str(int(round(yHat[i * 5 + j * 120 + k][0])))
                print(string, end=" ")
                if len(string) == 1:
                    print(" ", end="")
            print("\t\t", end="")
        print()
        if (i + 1) % 4 == 0:
            print()


def error_table():
    yHat = session.run(y, feed_dict={x: d.x_data, y_: d.y_data})
    yHat *= 50
    yGood = data.Data().y_data
    yGood *= 50
    for i in range(24):
        for j in range(4):
            for k in range(5):
                string = str(int(round(yHat[i * 5 + j * 120 + k][0] - yGood[i * 5 + j * 120 + k][0])))
                print(string, end=" ")
                if len(string) == 1:
                    print("  ", end="")
                elif len(string) == 2:
                    print(" ", end="")
            print("\t\t", end="")
        print()
        if (i + 1) % 4 == 0:
            print()


# Train!
train_network()

# Test
test_table()
error_table()
