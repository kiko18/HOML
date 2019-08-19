# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:59:54 2019

@author: BT
"""
'''
Using the tf estimater is good, but if you want more control over the architecture
of the network, you may prefer to use TF's lower-level Python API.
We will construct the same network as in mlp_estimator.py and we will implement
mini-batch GD to train it on MNIST
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

n_inputs = 28*28  # MNIST
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10


'''
we use placeholder node to represent the training data and target.
note that X and y here will be use for bach of training data and we don't
know yet how much instance will a batch containt.. 
So the shape of X is (None, n_feature) and the shape of y is None
'''

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


'''
Now we want to create the Neural Newtork.
-Note that the output layer use softmax activation function instead of Relu 
 softmax is generally is a good choice for classificatio task (when the class are mutually exclusive)
 For regression task, one ca use no activation function at all in the output layer.
-The two hidden layer are almost identical, they only differ by the inputs
 they are conected to and by the number of neuron they contain.
 In most cases, we can use Relu activation function (or its variant, like leaky relu) in the hidden layer, 
 it is a bit faster to compute than other activations and GD does not get stuck as much on plateaus,
 thanks to the fct that it does not saturate for large input values (as opposed to the logistic function of the tanh,
 which saturate at 1)
-The placeholder x will act as the input layer. During the execution phase, it will
 be replace with one trainning batch at a time (all the instance in a batch are processed
 simultaneously by the NN)
'''

'''
Create one layer at a time
params:
    -inputs
    -number of neuron
    -activation function
    name of the layer
'''
def neuron_layer(X, n_neurons, name, activation=None):
    #create a name scope with the name of the layer. It will containt all the 
    #computation nodes for this neuron layer. This is optional but the graph will
    #look much nicer in TensorBoard if its node are well organized.
    with tf.name_scope(name): 
        #get the number of input/features
        n_inputs = int(X.get_shape()[1])
        # initialise the weights W using a truncated normal (gaussian) distribution, with
        # a standard deviation of 2/sqrt(n_inputs + n_neurons)
        # Using this specific standart deviation help the algo converge much faster.
        # (we will discuss this in more details in later script, it is one of this small tweaks to 
        # neural networks that have had a tremendous impact on their efficiency).
        # moreover, using a truncated normal distribution rather than a regular normal 
        # distribution ensures that there won't be any large weights, which could slow down training.
        # W will be a 2D tensor containing all the connection weights 
        # between each input and each neuron, hence of shape (n_inputs, n_neurons).
        # It is important to initialize connections weight randomly for each hidden
        # layers to avoid any symmetries that GD algo would be unable to break
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="kernel")
        # initialize the bias variable with zero
        b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        # compute the weighted sum of the input and the bias for each and every neuron in the layer, 
        # for all instances in the bach in just one shot (vectorized implementation)
        #Note that ading 1D vec b to 2D matrix XW, add b to each column of XW, this is 
        #called broadcasting
        Z = tf.matmul(X, W) + b
        if activation is not None:
            return activation(Z)
        else:
            return Z


'''
create the Deep NN
'''
# once again we use a name scoope for clarity
with tf.name_scope("dnn"):
    # the first Hidden layer takes X as its input
    hidden1 = neuron_layer(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    # the second hidden layer takes the 1st HL as it input
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    # the output layer takes the last HL as its input
    # here as often, logits define the output of the network before going through the 
    # softmax activation function. We don't compute the softmax here because we will
    # incorporate it in the loss function
    logits = neuron_layer(hidden2, n_outputs, name="outputs") #output of NN before softmax is aplied
    
    
    
'''
Now that we have the NN model ready to go, we need to define the cost function that we will use to train it.
We will use the cross entropy. It penalize models that estimate a low probability for the target class. 
Tf provides several function to compute cross entropy. We will use sparse_softmax_cross_entropy_with_logits(),
it compute the cross entropy based on the "logits" (i.e, the output of the network before
going through the softmax activation function), 
and it expects labels in the form of integers ranging from 0 to n_classes-1 (in our case 0 to 9)
This give us a 1D tensor containing the cross entropy for each instance.
we can then use tf.reduce_mean() to compute the mean cross entropy over all instances.
'''    

with tf.name_scope("loss"):
    # Sparse_softmax_cross_entropy_with_logits is equivalent to applying softmax activation
    # Function and then computing the cross entropy, but it is more efficient.
    # It also properly takes care of corner cases like logits equal to 0.
    # There is also another fct called softmax_cross_entropy_with_logits(), which takes
    # labels in form of one-hot vectors (instead of ints from 0 to number_of_classes-1)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) #compute cross_entropy based on logit
    # compute the mean cross entropy over all instances.
    loss = tf.reduce_mean(xentropy, name="loss")
    

'''
We have the NN, we have the cost function, now we need to define a GradientDescentOptimizer
that will tweak the model parameters to minimize the cost function.
'''
learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)
    
'''
The last important step in the construction phase is to specify how to evaluate the model.
We will simply use accuracy as our performance measure.
First for each training instance, we determine if the neural network's prediction is correct
by checking whether or not the highest logit corresponds to the target class. 
This can be done using the in_top_k() fct, which return a 1D tensor full of boolean values,
so we need to cast these booleans to floats and then compute the average. This will give us the
Network overall accuracy
'''
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
 

'''
As usual we need to to create a node to initialize all variables
we also create a tf saver to save our trained model params to disk
'''

init = tf.global_variables_initializer()
saver = tf.train.Saver()


#--------Tensorboard--------------
# Create a node in the graph that will evaluate MSE and write it to a tensorBoard-compatible binary log string
#called summary. This is a way to log/debug training stats:
# we add a node to evaluate the MSE -> we evaluate it in the execution phase -> we store the result in a file -> 
# we can later display the file using tensorboard
loss_summary = tf.summary.scalar('loss', loss)

#create a FileWriter that we will use to write summaries to logfiles in the log directory
#First param = path to the log dir (defined at the begining of this script)
#second param= graph we want to visualize
#Note: the FileWriter create the log dir if it doesn't exist, and write the graph definition in 
#a binary logfile called an even file
logdir = "C:/Users/BT/Documents/others/tf/tf_boards/tf_logs/mlp_plain_tf" 
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph()) #write the default graph in the logdir folder
#--------------------------------

'''
Execution phase
'''
#load the MNIST dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

#show one training instance
plt.imshow(X_train[44, :, :])
plt.show()

#Scaling the data
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

#split the data into training and validation
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

n_epochs = 20
batch_size = 50
modelParamsDir = 'C:/Users/BT/Documents/others/tf/tf_boards/params/mlp_plain_tf.ckpt'

def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch



with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        batch_index = 0
        #at each epoch, we iterate through a number of mini-batches that correspond to the trainingset size.      
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            #we run the training operation, feeding it the current mini-batch input data and targets.
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
            #Evaluate the loss_summary node regulary during training. 
            #This output a summary that we can then write to the events file using file_wrier
            if(batch_index % 10) == 0:  #every 10 minibatch
                summary_str = loss_summary.eval(feed_dict = {X: X_batch, y: y_batch})
                file_writer.add_summary(summary_str) #add the summary to the event file
            batch_index +=1
            
        #At the end of each epoch we evaluates the model on the last mini-batch and on the full validation data
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print(epoch, "Batch accuracy:", acc_batch, "Val accuracy:", acc_val)
      
    #save model params after training on all epoch
    save_path = saver.save(sess, modelParamsDir)

#cd c:\programdata\anaconda3\lib\site-packages\tensorboard 
#python main.py --logdir=C:\Users\BT\Documents\others\tf\tf_boards\tf_logs\
file_writer.close()


'''
Use the network to make predictions
'''
with tf.Session() as sess:
    # load the model params from disk
    saver.restore(sess, modelParamsDir) # or better, use save_path
    # load some new images that we want to classify (it muss have been scaled the same way as the training data)
    X_new_scaled = X_test[:20]  
    # evaluate logit node (If you want to know all the estimated class proba, you need to apply the softmax(9 fct to the logits))
    #but if you just want to predict a class, you can just pick the class with the heighest logit value (argmax fct does the trick)
    Z = logits.eval(feed_dict={X: X_new_scaled}) 
    y_pred = np.argmax(Z, axis=1)
    