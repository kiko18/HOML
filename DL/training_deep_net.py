# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 09:26:25 2020

@author: BT
"""
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np


'''
In Chapter 10 we introduced artificial neural networks and trained our first deep
neural networks. But they were very shallow nets, with just a few hidden layers.
What if you need to tackle a very complex problem, such as detecting hundreds 
of types of objects in high-resolution images? You may need to train a much deeper DNN, 
perhaps with 10 layers or much more, each containing hundreds of neurons, connected
by hundreds of thousands of connections. This would not be a walk in the park:
- First, you would be faced with the tricky vanishing gradients problem (or the
  related exploding gradients problem) that affects deep neural networks and makes
  lower layers very hard to train.
- Second, you might not have enough training data for such a large network, or it
  might be too costly to label.
- Third, training may be extremely slow.
- Fourth, a model with millions of parameters would severely risk overfitting the
  training set, especially if there are not enough training instances, or they are too
  noisy.
In this chapter, we will go through each of these problems in turn and present techniques
to solve them.
'''


'''  
Vanishing/Exploding Gradients Problems
--------------------------------------
To summe up: 
    - Vanishing/expliding Gradient is the problem that the layers doesn't learn at the same speed
      some layers learn quickly while oder stay virtually unchanged after gradient update.
    - Glorot and bengio showed in 2010 that this problem is due to a combination of poor random 
      initialisation and (at that time popular) sigmoid activation function. which cause the
      variance at the output of each layers to be more greater than the variance of its input.
    - To solve this problem, they propose a weight initialization method called Xavier 
      initialization. Which enssure that the variance of the outputs of each layer is equal to the 
      variance of its inputs. Moreover, it ensure that the gradients has equal variance before and 
      after flowing through a layer in the reverse direction. It is actually not possible to 
      guarantee both unless the layer has an equal number of inputs (fan-in) and neurons (fan-out),
      but they proposed a good compromise that has proven to work very well in practice.
      
Detailled version:      
As we discussed in Chapter 10, the backpropagation algorithm works by going from
the output layer to the input layer, propagating the error gradient on the way. 
Once the algorithm has computed the gradient of the cost function with regards to each
parameter in the network, it uses these gradients to update each parameter with a
Gradient Descent step.
Unfortunately, gradients often get smaller and smaller as the algorithm progresses
down to the lower layers. As a result, the Gradient Descent update leaves the lower
layer connection weights virtually unchanged, and training never converges to a good
solution. This is called the vanishing gradients problem. In some cases, the opposite
can happen: the gradients can grow bigger and bigger, so many layers get insanely
large weight updates and the algorithm diverges. This is the exploding gradients problem,
which is mostly encountered in recurrent neural networks. More generally,
deep neural networks suffer from unstable gradients; different layers may learn at
widely different speeds.
Although this unfortunate behavior has been empirically observed for quite a while
(it was one of the reasons why deep neural networks were mostly abandoned for a
long time), it is only around 2010 that significant progress was made in understanding it. 
A paper titled “Understanding the Difficulty of Training Deep Feedforward Neural Networks” 
by Xavier Glorot and Yoshua Bengio found a few suspects, including the combination of the 
popular logistic sigmoid activation function and the weight initialization technique that was 
most popular at the time, namely random initialization using a normal distribution with a 
mean of 0 and a standard deviation of 1.
In short, they showed that with this activation function and this initialization scheme,
the variance of the outputs of each layer is much greater than the variance of its
inputs. Going forward in the network, the variance keeps increasing after each layer
until the activation function saturates at the top layers. This is actually made worse by
the fact that the logistic function has a mean of 0.5, not 0 (the hyperbolic tangent
function has a mean of 0 and behaves slightly better than the logistic function in deep
networks).
Looking at the logistic activation function, you can see that when inputs become large 
(negative or positive), the function saturates at 0 or 1, with a derivative extremely close to 0. 
Thus when backpropagation kicks in, it has virtually no gradient to propagate back through the 
network, and what little gradient exists keeps getting diluted as backpropagation progresses 
down through the top layers, so there is really nothing left for the lower layers.

Glorot and He Initialization
----------------------------
In their paper, Glorot and Bengio propose a way to significantly alleviate this problem.
We need the signal to flow properly in both directions: in the forward direction
when making predictions, and in the reverse direction when backpropagating gradients.
We don’t want the signal to die out, nor do we want it to explode and saturate.
For the signal to flow properly, the authors argue that we need the variance of the
outputs of each layer to be equal to the variance of its inputs, and we also need the
gradients to have equal variance before and after flowing through a layer in the
reverse direction (please check out the paper if you are interested in the mathematical
details). It is actually not possible to guarantee both unless the layer has an equal
number of inputs (fan-in) and neurons (fan-out), but they proposed a good compromise that has proven 
to work very well in practice: the connection weights of each layer must be initialized randomly 
such that they have: 
    - Normal distribution with mean 0 
    - Normal distribution with variance σ^2 = 1/fan_avg with fan_avg = (fan_in + fan_out) /2.
Using Glorot initialization can speed up training considerably, and it is one of the tricks 
that led to the current success of Deep Learning.
'''









'''
Transfer Learning
-----------------
It is generally not a good idea to train a very large DNN from scratch: instead, you
should always try to find an existing neural network that accomplishes a similar task
to the one you are trying to tackle (we will discuss how to find them in Chapter 14),
then just reuse the lower layers of this network: this is called transfer learning. 
It will not only speed up training considerably, but will also require much less training data.
For example, suppose that you have access to a DNN that was trained to classify pictures
into 100 different categories, including animals, plants, vehicles, and everyday
objects. You now want to train a DNN to classify specific types of vehicles. These
tasks are very similar, even partly overlapping, so you should try to reuse parts of the
first network.

If the input pictures of your new task don’t have the same size as
the ones used in the original task, you will usually have to add a
preprocessing step to resize them to the size expected by the original model. 
More generally, transfer learning will work best when the inputs have similar low-level features.

The output layer of the original model should usually be replaced since it is most
likely not useful at all for the new task, and it may not even have the right number of
outputs for the new task.
Similarly, the upper hidden layers of the original model are less likely to be as useful
as the lower layers, since the high-level features that are most useful for the new task
may differ significantly from the ones that were most useful for the original task. 
Therefore, You want to find the right number of layers to reuse.
The more similar the tasks are, the more layers you want to reuse
(starting with the lower layers). For very similar tasks, you can try
keeping all the hidden layers and just replace the output layer.

Try freezing all the reused layers first (i.e., make their weights non-trainable, so gradient
descent won’t modify them), then train your model and see how it performs.
Then try unfreezing one or two of the top hidden layers to let backpropagation tweak
them and see if performance improves. The more training data you have, the more
layers you can unfreeze. It is also useful to reduce the learning rate when you unfreeze
reused layers: this will avoid wrecking their fine-tuned weights.
If you still cannot get good performance, and you have little training data, try dropping
the top hidden layer(s) and freeze all remaining hidden layers again. You can
iterate until you find the right number of layers to reuse. If you have plenty of training
data, you may try replacing the top hidden layers instead of dropping them, and
even add more hidden layers.
'''

# Load fashion Mnist and create a validation set
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

#Let's split the fashion MNIST training set in two:
#X_train_A: all images of all items except for sandals and shirts (classes 5 and 6).
#X_train_B: a much smaller training set of just the first 200 images of sandals or shirts.
#The validation set and the test set are also split this way, but without restricting the number of images.
#We will train a model on set A (classification task with 8 classes), and try to reuse it to tackle set B (binary classification). 
#We hope to transfer a little bit of knowledge from task A to task B, since classes in set A (sneakers, ankle boots, coats, t-shirts, etc.) 
#are somewhat similar to classes in set B (sandals and shirts). However, since we are using Dense layers, only patterns that occur 
#at the same location can be reused (in contrast, convolutional layers will transfer much better, since learned patterns can be detected 
#anywhere on the image, as we will see in the CNN chapter).


def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6) # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2 # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32) # binary classification task: is it a shirt (class 6)?
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))

(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]

''' Train model A '''
model_A = keras.models.Sequential()
model_A.add(keras.layers.Flatten(input_shape=[28, 28]))

for n_hidden in (300, 100, 50, 50, 50):
    model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
    
model_A.add(keras.layers.Dense(8, activation="softmax"))


model_A.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=["accuracy"])

print("Training model A")
history = model_A.fit(X_train_A, y_train_A, 
                      epochs=20,
                      validation_data=(X_valid_A, y_valid_A))

model_A.save("my_model_A.h5")


''' Train model B '''
model_B = keras.models.Sequential()
model_B.add(keras.layers.Flatten(input_shape=[28, 28]))

for n_hidden in (300, 100, 50, 50, 50):
    model_B.add(keras.layers.Dense(n_hidden, activation="selu"))
model_B.add(keras.layers.Dense(1, activation="sigmoid"))

model_B.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=["accuracy"])

print("Training model B")
history = model_B.fit(X_train_B, y_train_B, 
                      epochs=20,
                      validation_data=(X_valid_B, y_valid_B))



''' Reuse weight of model A to train Model B'''

model_A = keras.models.load_model("my_model_A.h5")              # load model A
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])     # create a new model, add all layers of A except the output layer
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))   # add an output layer, which will be initialized randomly

#Doing this way,  model_A and model_B_on_A now share some layers. 
#When you train model_B_on_A, it will also affect model_A. 
#If you want to avoid that, you need to clone model_A before you reuse its layers. 
#To do this, you must clone model A’s architecture, 
#then copy its weights (since clone_model() does not clone the weights).
#   model_A_clone = keras.models.clone_model(model_A)
#   model_A_clone.set_weights(model_A.get_weights())


#Now we could just train model_B_on_A for task B, but since the new output layer was
#initialized randomly, it will make large errors, at least during the first few epochs, so
#there will be large error gradients that may wreck the reused weights. To avoid this,
#one approach is to freeze the reused layers during the first few epochs, giving the new
#layer some time to learn reasonable weights. To do this, simply set every layer’s 
#trainable attribute to False and compile the model
for layer in model_B_on_A.layers[:-1]:  #freeze all layer except the last layer
    layer.trainable = False

# You must always compile your model after you freeze or unfreeze layers.
model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(lr=1e-3),   #compile using stochastic gradient descent
                     metrics=["accuracy"])

#train on 4 epochs to give the new layer (output layer) some time to learn reasonable weights
print("Training model model_B_on_A with froozen weights to give the new time to learn reasonable weighs")
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,          
                           validation_data=(X_valid_B, y_valid_B))

# Now we can unfreeze the reused layers (which requires compiling the model again) 
# and continue training to fine-tune the reused layers for task B. 
# After unfreezing the reused layers, it is usually a good idea to reduce
#the learning rate, once again to avoid damaging the reused weights
for layer in model_B_on_A.layers[:-1]:  #unfreeze all layer except the last layer(which is already new)
    layer.trainable = True

# recompile the model (always after freez or unfreez)
model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(lr=1e-3),
                     metrics=["accuracy"])

# train 
print("Training model model_B_on_A with unfroozen weights")
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_valid_B, y_valid_B))



