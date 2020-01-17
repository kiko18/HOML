#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:59:55 2019

@author: bt
"""

'''
It is not generally a good idea to train a very large DNN from scratch: instead, 
you should always try to find an existing neural network that accomplishes a similar task
to the one you are trying to tackle, then just reuse the lower layers of this network:
this is called transfer learning. It will not only speed up trainning considerably but will
also require much less data.
For picture for example, the if your new task don't have the same size as the ones used in the original task,
you will have to add a preprocessing step to rezise them to the size excepted by the original model.
More generally, transfer learning will only work well if the inputs have similar low-level features.
'''
import tensorflow as tf
import numpy as np

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
'''
To restore a model we load the graph's structure. The import_meta_graph() function does just that, 
loading the graph's operations into the default graph, and returning a Saver that 
you can then use to restore the model's state. 
We will restore the model trained in batch_norm.py
'''
saver = tf.train.import_meta_graph("./mylogs/tf_models/my_batch_norm.ckpt.meta")

'''
After exporting the graph we must get a handle on the operations and tensors we will need for training.
The best way to found out which operation we need is explore the graph using tensorboard, 
for this you must first export the graph using a FileWriter as seen in batch_norm.py
Once you know which operations you need, you can get a handle on them using the graph's get_operation_by_name() 
or get_tensor_by_name() methods:
'''
#X = tf.get_default_graph().get_tensor_by_name("X:0")
#y = tf.get_default_graph().get_tensor_by_name("y:0")
#accuracy = tf.get_default_graph().get_tensor_by_name("eval/accuracy:0")    #TODO: find out how to name this operation in batch_norm.py
#training_op = tf.get_default_graph().get_operation_by_name("GradientDescent")#TODO: find out how to name this operation in batch_norm.py
'''
If you're the author of the original model, you could make things easier for people who will reuse your model by giving operations
very clear names and documenting them. Another approach is to create a collection containing all the important operations that people
will want to get a handle on, as done in batch_norm.py line 116.
This way the operation can simply be load using: X, y, accuracy, training_op = tf.get_collection("my_important_ops")
'''
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
        

#the is a problem, tf is loading the operation twice
operations = tf.get_collection("my_important_ops")
X = operations[0]
y = operations[1]
accuracy = operations[2]
training_op = operations[3]
        
# start a session, restore the model's state and continue training on your data:      
with tf.Session() as sess:
    saver.restore(sess, "./mylogs/tf_models/my_batch_norm.ckpt")
    
    # continue training the model...
    print(" \n ------ \n continue training a restored model...: \n ------ \n ")
    
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print("epoch", epoch, "Validation accuracy:", accuracy_val)

    save_path = saver.save(sess, "./mylogs/tf_models/my_further_trained_batch_norm.ckpt")



'''
Note that, the more similar the tasks are, the more layers you want to reuse (starting from lower layers).
For very similar tasks, you can try keeping all the hidden layers and just replace the output layer.
'''
import sys
sys.modules[__name__].__dict__.clear()







