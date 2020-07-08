import tensorflow as tf
from matplotlib import pyplot as plt
from functools import partial
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#Auto encoder with a constraint on hidden layer, sparcity loss with kl divergence
#Prob distribution for kl divergence is fixed 

use_sparcity_loss = True
sparcity_weight = 0.1
sparcity_loss = 0.1

def kl_divergence(p, q):
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))


n_inputs = 28 * 28  # for MNIST
n_hidden1 = 300
n_hidden2 = 100  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
training = tf.placeholder_with_default(False, shape=(), name='training')


he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.elu,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)

hidden1 = my_dense_layer(X, n_hidden1,activation=tf.nn.sigmoid)
hidden2 = my_dense_layer(hidden1, n_hidden2)  # codings
hidden3 = my_dense_layer(hidden2, n_hidden3)
outputs = my_dense_layer(hidden3, n_outputs, activation=None)

# from autoencode_coupled import coupled_net
# outputs, X = coupled_net(n_inputs,n_hidden1,n_hidden2,n_hidden3,n_outputs,l2_reg)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  # MSE


reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)\


loss = tf.add_n([reconstruction_loss] + reg_losses)
sparc_loss = None
if(use_sparcity_loss):
    mean_acivation = tf.math.reduce_mean(hidden1,axis=0)
    sparc_loss = tf.reduce_mean(kl_divergence(sparcity_weight,mean_acivation))
    loss = loss + tf.where(training,sparcity_loss*sparc_loss,0)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


# n_epochs = 1#5
# batch_size = 50000#150

n_epochs = 3
batch_size = 150

n_test_digits = 4
X_test = mnist.test.images[:n_test_digits]

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def test_images(outputs_val):
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])
        plt.savefig('sparse.png')

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            _, l, o = sess.run([training_op,loss,outputs], feed_dict={X: X_batch,training: True})
            print(l)

    print('test loss train')
    print(loss.eval(feed_dict={X: X_test,training: True}))
    print('test loss test')
    print(loss.eval(feed_dict={X: X_test,training: False}))

    outputs_val = outputs.eval(feed_dict={X: X_test})
    print(loss.eval(feed_dict={X: X_test}))
    test_images(outputs_val)


