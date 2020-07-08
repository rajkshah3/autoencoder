import tensorflow as tf
from matplotlib import pyplot as plt
from functools import partial
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_inputs = 28 * 28  # for MNIST
n_hidden1 = 300
n_hidden2 = 100  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
# my_dense_layer = partial(tf.layers.dense,
#                          activation=tf.nn.elu,
#                          kernel_initializer=he_init,
#                          kernel_regularizer=l2_regularizer)

# hidden1 = my_dense_layer(X, n_hidden1)
# hidden2 = my_dense_layer(hidden1, n_hidden2)  # codings
# hidden3 = my_dense_layer(hidden2, n_hidden3)
# outputs = my_dense_layer(hidden3, n_outputs, activation=None)

from autoencode_coupled import coupled_net
outputs, X, weights, hidden_output1 = coupled_net(n_inputs,n_hidden1,n_hidden2,n_hidden3,n_outputs,l2_reg)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  # MSE


reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


reconstruction_loss2 = tf.reduce_mean(tf.square(outputs_2 - X))  # MSE
loss2 = tf.add_n([reconstruction_loss2] + reg_losses)
training_op_2 = optimizer.minimize(loss2)

# n_epochs = 1#5
# batch_size = 50000#150

n_epochs = 5
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            _, l, o = sess.run([training_op,loss,outputs], feed_dict={X: X_batch})
            print(l)

    n_test_digits = 4
    X_test = mnist.test.images[:n_test_digits]

    outputs_val = outputs.eval(feed_dict={X: X_test})

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

for digit_index in range(n_test_digits):
    plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
    plot_image(X_test[digit_index])
    plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
    plot_image(outputs_val[digit_index])
    plt.savefig('foo.png')
    input()


