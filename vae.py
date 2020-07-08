import tensorflow as tf
from matplotlib import pyplot as plt
from functools import partial
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

n_inputs = 28 * 28  # for MNIST
n_hidden1 = 500
n_hidden2 = 500 # 
coding_size = 10 # codings
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.001
l2_reg = 0.00001

def sample(log_var,mean):
    # return tf.random.normal(tf.shape(log_var),mean,tf.math.exp(log_var / 2))
    return tf.random.normal(tf.shape(log_var))* tf.exp(0.5*log_var) + mean

def calc_sample_loss(log_var,mean):
    return -0.5*tf.reduce_sum(log_var + 1 - tf.exp(log_var) - mean*mean)

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

he_init = tf.contrib.layers.variance_scaling_initializer()
l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.elu,
                         kernel_initializer=he_init,
                         kernel_regularizer=l2_regularizer)

hidden1 = my_dense_layer(X, n_hidden1,activation=tf.nn.sigmoid)
hidden2 = my_dense_layer(hidden1, n_hidden2)  # codings
mean = my_dense_layer(hidden2,coding_size,activation=None)
log_variance = my_dense_layer(hidden2,coding_size,activation=None)
middle  = sample(log_variance,mean)

hidden4 = my_dense_layer(middle, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5)
outputs = my_dense_layer(hidden5, n_outputs, activation=None)

# from autoencode_coupled import coupled_netc
# outputs, X = coupled_net(n_inputs,n_hidden1,n_hidden2,n_hidden3,n_outputs,l2_reg)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))  # MSE

sample_loss = calc_sample_loss(log_variance,mean)/(n_inputs*200)
reg_losses = [tf.constant(0,dtype=tf.float32)] #tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses + [sample_loss])
# loss = tf.add_n([reconstruction_loss] + reg_losses)

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")

def test_images(outputs_val,name='foo'):
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])
    
    plt.savefig('{}.png'.format(name))
    plt.clf()
# n_epochs = 1#5
# batch_size = 50000#150

n_epochs = 50
batch_size = 250

n_test_digits = 4
X_test = mnist.test.images[:n_test_digits]

n_digits = 8

with tf.Session() as sess:
    init.run()
    def train(n_epochs):
        for epoch in range(n_epochs):
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):
                X_batch, y_batch = mnist.train.next_batch(batch_size)
                _, l,l1,l2, o = sess.run([training_op,loss,reconstruction_loss,sample_loss,outputs], feed_dict={X: X_batch})
                print(l)
                print('rec_loss: {}, sample loss: {}'.format(l1,l2))
    train(n_epochs)

    outputs_val = outputs.eval(feed_dict={X: X_test})
    print(loss.eval(feed_dict={X: X_test}))
    test_images(outputs_val,'vae')

    # import pdb; pdb.set_trace()  # breakpoint 5658d980 //

    codings_rnd = np.random.normal(size=[n_digits, coding_size])
    outputs_val = outputs.eval(feed_dict={middle: codings_rnd})

    
    for iteration in range(n_digits):
        plt.subplot(n_digits/2, 2, iteration + 1)
        plot_image(outputs_val[iteration])
    plt.savefig('{}.png'.format('generated_vae'))
    plt.clf()
    
    print('done')

