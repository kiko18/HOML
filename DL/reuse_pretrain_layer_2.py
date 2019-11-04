# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:40:13 2019

@author: BT
"""


import tensorflow as tf
import numpy as np

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# Now you can start a session, restore the model's state and continue training on your data:
# Let first load the data and define shuffle_batch function
# Load the data
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_epochs = 20
batch_size = 200


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
        

'''
In general you will want to reuse only the lower layers. If you are using import_meta_graph() it will load the whole graph, 
but you can simply ignore the parts you do not need. In this example, we add a new 3th and 4th hidden layer on top of the pretrained 
1rd and 2nd layer (ignoring the old 3th and 4th hidden layer). We also build a new output layer, the loss for this new output, 
and a new optimizer to minimize it. We also need another saver to save the whole graph (containing both the entire old graph
plus the new operations), and an initialization operation to initialize all the new variables:
'''

reset_graph()

n_hidden3 = 65  # new layer
n_hidden4 = 25  # new layer
n_outputs = 10  # new layer
learning_rate = 0.01 

saver = tf.train.import_meta_graph("./mylogs/tf_models/my_batch_norm.ckpt.meta")

X = tf.get_default_graph().get_tensor_by_name("X:0")
y = tf.get_default_graph().get_tensor_by_name("y:0")

hidden1 = tf.get_default_graph().get_tensor_by_name("dnn/bn1_act:0")
hidden2 = tf.get_default_graph().get_tensor_by_name("dnn/bn2_act:0")

new_hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="new_hidden3")
new_hidden4 = tf.layers.dense(new_hidden3, n_hidden4, activation=tf.nn.relu, name="new_hidden4")
new_logits = tf.layers.dense(new_hidden4, n_outputs, name="new_outputs")

with tf.name_scope("new_loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=new_logits)
    loss = tf.reduce_mean(xentropy, name="loss")

with tf.name_scope("new_eval"):
    correct = tf.nn.in_top_k(new_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

with tf.name_scope("new_train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
new_saver = tf.train.Saver()

# now we can train the model
with tf.Session() as sess:
    init.run()
    saver.restore(sess, "./mylogs/tf_models/my_batch_norm.ckpt")
    
    print(" \n ------ \n continue training a restored model while changing the uppers layer: \n ------ \n ")
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print("epoch", epoch, "Validation accuracy:", accuracy_val)

    save_path = new_saver.save(sess, "./mylogs/tf_models/my_retrained_batch_norm.ckpt")


'''
Note that, the more similar the tasks are, the more layers you want to reuse (starting from lower layers).
For very similar tasks, you can try keeping all the hidden layers and just replace the output layer.
'''









