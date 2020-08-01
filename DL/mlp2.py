# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:55:33 2020

@author: BT
"""

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np
import math

# First, we need to load a dataset. We will tackle Fashion MNIST. It has the exact same format as
# MNIST (70,000 grayscale images of 28×28 pixels each, with 10 classes), but the
# images represent fashion items rather than handwritten digits, so each class is more
# diverse and the problem turns out to be significantly more challenging than MNIST.
# For example, a simple linear model reaches about 92% accuracy on MNIST, but only
# about 83% on Fashion MNIST.


# When loading MNIST or Fashion MNIST using Keras rather than Scikit-Learn, one
# important difference is that every image is represented as a 28×28 array rather than a
# 1D array of size 784. Moreover, the pixel intensities are represented as integers (from
# 0 to 255) rather than floats (from 0.0 to 255.0).
                                                                                   
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("shape of X_train_full = ", X_train_full.shape)
print("shape of X_train_full = ", X_train_full.dtype)

# With MNIST, when the label is equal to 5, it means that the image represents the
# handwritten digit 5. Easy. However, for Fashion MNIST, we need the list of class
# names to know what we are dealing with
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# create a validation set
# remember that Gradient Descent works better if the feature are scaled
# We scale the feature (pixel intensities) down to the 0-1 range by dividing them by 255.0 (this also
# converts them to floats)
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0


img_id = 15
plt.imshow(X_train[img_id], cmap="binary")
plt.axis('off')
plt.title(class_names[y_train[img_id]])
plt.show()



del X_train_full, y_train_full



'''
Building a Classification MLP Using the Sequential API
-----------------------------------------------------
We will see how to build, train, evaluate and use a classification MLP using the Sequential API
'''
# Now let’s build the neural network! Here is a classification MLP with two hidden layers:
model = keras.models.Sequential()
# Add a flatten layer whose role is simply to convert each input image into a 1D array
# if it receives input data X, it computes X.reshape(-1, 1). This layer does not have any parameters, 
#it is just there to do some simple preprocessing. Since it is the first layer in the model,
# you should specify the input_shape
model.add(keras.layers.Flatten(input_shape=[28, 28]))
# Next, we add 2 dense layers. Each Dense layer manages its own weight matrix, containing all the
# connection weights between the neurons and their inputs. It also manages a vecIt or of bias terms 
#(one per neuron). When it receives some input data, it computes g(XW+b)
model.add(keras.layers.Dense(300, activation="relu")) 
model.add(keras.layers.Dense(100, activation="relu"))
# Finally, we add a Dense output layer with 10 neurons (one per class), using the
# softmax activation function (because the classes are exclusive).
model.add(keras.layers.Dense(10, activation="softmax"))

# Instead of adding the layer one by one, we can just pass a list of layer when creating the model.
keras.backend.clear_session()
model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28, 28]),
keras.layers.Dense(300, activation="relu", name="hiden_1"),
keras.layers.Dense(100, activation="relu", name="hiden_2"),
keras.layers.Dense(10, activation="softmax", name="softmax")    #sigmoid for binary classification
])

model.summary()
keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)
# Note that Dense layers often have a lot of parameters. For example, the first hidden
#layer has 300 x 784 connection weights, plus 300 bias terms, which adds up to
#235,500 parameters! This gives the model quite a lot of flexibility to fit the training
#data, but it also means that the model runs the risk of overfitting, especially when you
#do not have a lot of training data. We will come back to this later. 
print(model.layers)

hidden1 = model.layers[1]
weights, biases = hidden1.get_weights()
# Notice that the Dense layer initialized the connection weights randomly (which is
# needed to break symmetry), and the biases were just initializedto zeros, which is fine

# After a model is created, you must call its compile() method to specify the loss function
# and the optimizer to use. Optionally, you can also specify a list of extra metrics to
# compute during training and evaluation:
model.compile(loss="sparse_categorical_crossentropy",   #keras.losses.sparse_categorical_crossentropy
              optimizer="sgd",  # or keras.optimizers.SGD()
              metrics=["accuracy"]) #only for classifier keras.metrics.sparse_categorical_accuracy

# we use the "sparse_categorical_crossentropy" loss because we have sparse labels 
# (i.e., for each instance there is just a target class index, from 0 to 9 in this case), 
# and the classes are exclusive. If instead we had one target probability per class for each instance 
# (such as one-hot vectors, e.g. [0.,0., 0., 1., 0., 0., 0., 0., 0., 0.] to represent class 3), 
# then we would need to use the "categorical_crossentropy" loss instead. 
# If we were doing binary classification (with one or more binary labels), then we would use the "sigmoid" 
# (i.e., logistic) activation function in the output layer instead of the "softmax" activation
# function, and we would use the "binary_crossentropy" loss.
history = model.fit(X_train, y_train, 
                    epochs=20, #30
                    validation_data=(X_valid, y_valid)  #Keras measure loss and metrics on this set at theend of each epoch
                    #Instead of passing a validation set using the validation_data
                    #argument, you could instead set validation_split to the ratio of
                    #the training set that you want Keras to use for validation (e.g., 0.1).
                    )

'''
If the training set was very skewed, with some classes being overrepresented and others
underrepresented, it would be useful to set the class_weight argument when
calling the fit() method, giving a larger weight to underrepresented classes, and a
lower weight to overrepresented classes. These weights would be used by Keras when
computing the loss. If you need per-instance weights instead, you can set the sam
ple_weight argument (it supersedes class_weight). This could be useful for example
if some instances were labeled by experts while others were labeled using a
crowdsourcing platform: you might want to give more weight to the former. You can
also provide sample weights (but not class weights) for the validation set by adding
them as a third item in the validation_data tuple.
''' 

'''
The fit() method returns a History object containing the training parameters (his
tory.params), the list of epochs it went through (history.epoch), and most importantly
a dictionary (history.history) containing the loss and extra metrics it
measured at the end of each epoch on the training set and on the validation set (if any).
This can be used to plot the learning curve
'''
print('\n history params: \n', history.params)
print('\n history is a dict with keys: \n', history.history.keys())
print(' \n history loss: \n', history.history['loss'])


plt.plot(history.epoch, history.history['loss'], label="train loss")
plt.plot(history.epoch, history.history['accuracy'], label="train accuracy")
plt.plot(history.epoch, history.history['val_loss'], label="validation loss")
plt.plot(history.epoch, history.history['val_accuracy'], label="validation accuracy")
plt.grid(True)
plt.title('mondel trained on Fashion mnist')
plt.legend()
plt.show()


# plot the learning curve
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('mondel trained on Fashion mnist')
plt.show()

'''
You can see that both the training and validation accuracy steadily increase during
training, while the training and validation loss decrease. Good! Moreover, the validation
curves are quite close to the training curves, which means that there is not too
much overfitting. In this particular case, the model performed better on the validation
set than on the training set at the beginning of training: this sometimes happens
by chance (especially when the validation set is fairly small). However, the training set
performance ends up beating the validation performance, as is generally the case
when you train for long enough. 
You can tell that the model has not quite converged
yet, as the validation loss is still going down, so you should probably continue training.
It’s as simple as calling the fit() method again, since Keras just continues training
where it left off (you should be able to reach close to 89% validation accuracy).
'''

'''
If you are not satisfied with the performance of your model, you should go back and
tune the model’s hyperparameters, for example the number of layers, the number of
neurons per layer, the types of activation functions we use for each hidden layer, the
number of training epochs, the batch size (it can be set in the fit() method using the
batch_size argument, which defaults to 32). We will get back to hyperparameter
tuning at the end of this chapter.

Once you are satisfied with your model’s validation accuracy, you should evaluate it 
on the test set to estimate the generalization error before you deploy the model to production. 
You can easily do this using the evaluate() method (it also supports several other arguments, 
such as batch_size or sample_weight, please check the documentation for more details):
'''
model.evaluate(X_test, y_test)

'''
As we saw in Chapter 2, it is common to get slightly lower performance on the test set
than on the validation set, because the hyperparameters are tuned on the validation
set, not the test set (however, in this example, we did not do any hyperparameter tuning,
so the lower accuracy is just bad luck). Remember to resist the temptation to
tweak the hyperparameters on the test set, or else your estimate of the generalization
error will be too optimistic.
'''

#we can use the model’s predict() method to make predictions on new instances.
print('---------------------------------------')
img_id = 30
test_img = X_test[img_id,]
plt.imshow(test_img, cmap="binary"); plt.show()
test_img_exp = np.expand_dims(test_img, axis=0)
predicted=model.predict(test_img_exp)
predicted_class = np.argmax(predicted)
print('True label :', class_names[y_test[img_id]])
print('predicted :', class_names[predicted_class], ' :with proba', np.max(predicted))
print('---------------------------------------')

'''
The model “believes” x_new is probably ankle boots, but it’s not entirely sure, 
it might be sneakers instead. If you only care about the class with the highest estimated
probability (even if that probability is quite low) then you can use the predict_classes() method instead.
'''
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))
y_pred = model.predict_classes(X_new)
np.array(class_names)[y_pred]


# Visualize the weights of the first hidden layer.
hidden1 = model.layers[1]
weights0, biases = hidden1.get_weights()
num_nodes = weights0.shape[1]
num_rows = int(math.ceil(num_nodes / 10.0))
fig, axes = plt.subplots(num_rows, 10, figsize=(20, 2 * num_rows))
for coef, ax in zip(weights0.T, axes.ravel()):
    # Weights in coef is reshaped from 1x784 to 28x28.
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.pink)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()


'''
Building a Regression MLP Using the Sequential API
--------------------------------------------------

Let’s switch to the California housing problem and tackle it using a regression neural
network. For simplicity, we will use Scikit-Learn’s fetch_california_housing()
function to load the data: this dataset is simpler than the one we used in Chapter 2,
since it contains only numerical features (there is no ocean_proximity feature), and
there is no missing value.
'''
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()

'''
we can use pandas to better analyse the data
'''
import pandas as pd
df = pd.DataFrame(data=housing.data, columns= housing.feature_names)
df.hist(bins=50, figsize=(20,15))

X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)

del X_train_full, y_train_full

# scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

'''
Building, training, evaluating and using a regression MLP using the Sequential API to
make predictions is quite similar to what we did for classification. The main differences
are the fact that the output layer has a single neuron (since we only want to
predict a single value) and uses no activation function, and the loss function is the
mean squared error. Since the dataset is quite noisy, we just use a single hidden layer
with fewer neurons than before, to avoid overfitting:
'''
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1)
])

model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]
y_pred = model.predict(X_new)

plt.plot(pd.DataFrame(history.history))
plt.grid(True)
#plt.gca().set_ylim(0, 1)
plt.show()


'''
Building Complex Models Using the Functional API
-----------------------------------------------
'''
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)     #hidden1 takes input_ as input
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)    #hiden2 takes hidden1 as input
concat = keras.layers.concatenate([input_, hidden2])            #concat takes input_ and hidden2 as input
output = keras.layers.Dense(1)(concat)                          #output takes concat as input
model = keras.models.Model(inputs=[input_], outputs=[output])   #create a model specifiant the input & outp

model.summary()

model.compile(loss="mean_squared_error", optimizer=keras.optimizers.SGD(lr=1e-3))
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
y_pred = model.predict(X_new)

# MULTIPLE INPUT
# what if you want to send a subset of the features through the wide path, and a
#different subset (possibly overlapping) through the deep path (see Figure 10-14)? In
#this case, one solution is to use multiple inputs. For example, suppose we want to
#send 5 features through the deep path (features 0 to 4), and 6 features through the
#wide path (features 2 to 7)
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])


model.compile(loss="mse", optimizer="sgd")

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]

# when we call the fit() method, instead of passing a single input matrix X_train, 
#we must pass a pair of matrices (X_train_A, X_train_B): one per input. 
# The same is true for X_valid, and also for X_test and X_new when you call evaluate() or predict()
history = model.fit((X_train_A, X_train_B), y_train, epochs=20,
                    validation_data=((X_valid_A, X_valid_B), y_valid))
mse_test = model.evaluate((X_test_A, X_test_B), y_test)
y_pred = model.predict((X_new_A, X_new_B))

# MULTIPLE OUTPUTS
# There are also many use cases in which you may want to have multiple outputs:
# -  The task may demand it, for example you may want to locate and classify the
#    main object in a picture. This is both a regression task (finding the coordinates of
#the object’s center, as well as its width and height) and a classification task.
#  - Similarly, you may have multiple independent tasks to perform based on the
#    same data. Sure, you could train one neural network per task, but in many cases
#    you will get better results on all tasks by training a single neural network with
#    one output per task. This is because the neural network can learn features in the
#    data that are useful across tasks.
#  - Another use case is as a regularization technique (i.e., a training constraint whose
#    objective is to reduce overfitting and thus improve the model’s ability to generalize).
#For example, you may want to add some auxiliary outputs in a neural network
#architecture (see Figure 10-15) to ensure that the underlying part of the
#network learns something useful on its own, without relying on the rest of the network
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.models.Model(inputs=[input_A, input_B],
                           outputs=[output, aux_output])

# Each output will need its own loss function
# Since we care much more about the main output than about the auxiliary output 
#(as it is just used for regularization), so we want to give the main output’s loss a much greater weight. 
# Fortunately, it is possible to set all the loss weights when compiling the mode
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1], optimizer=keras.optimizers.SGD(lr=1e-3))


# When we train the model, we need to provide some labels for each output. 
# In this example, the main output and the auxiliary output should try to predict the same
# thing, so they should use the same labels. So instead of passing y_train, we just need
# to pass (y_train, y_train)
history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
                    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))


# When we evaluate the model, Keras will return the total loss, as well as all the individual losses
total_loss, main_loss, aux_loss = model.evaluate([X_test_A, X_test_B], [y_test, y_test])
y_pred_main, y_pred_aux = model.predict([X_new_A, X_new_B])

'''
Building Dynamic Models Using the Subclassing API
-------------------------------------------------
Both the Sequential API and the Functional API are declarative: you start by declaring
which layers you want to use and how they should be connected, and only then
can you start feeding the model some data for training or inference. This has many
advantages: the model can easily be saved, cloned, shared, its structure can be displayed
and analyzed, the framework can infer shapes and check types, so errors can
be caught early (i.e., before any data ever goes through the model). It’s also fairly easy
to debug, since the whole model is just a static graph of layers. But the flip side is just
that: it’s static. Some models involve loops, varying shapes, conditional branching,
and other dynamic behaviors. For such cases, or simply if you prefer a more imperative
programming style, the Subclassing API is for you.
Simply subclass the Model class, create the layers you need in the constructor, and use
them to perform the computations you want in the call() method. For example, creating
an instance of the following WideAndDeepModel class gives us an equivalent
model to the one we just built with the Functional API. You can then compile it, evaluate
it and use it to make predictions, exactly like we just did.
'''
class WideAndDeepModel(keras.models.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
        
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel(30, activation="relu")

'''
This example looks very much like the Functional API, except we do not need to create
the inputs, we just use the input argument to the call() method, and we separate
the creation of the layers in the constructor from their usage in the call() method.
However, the big difference is that you can do pretty much anything you want in the
call() method: for loops, if statements, low-level TensorFlow operations, your
imagination is the limit (see Chapter 12)! This makes it a great API for researchers
experimenting with new ideas.
However, this extra flexibility comes at a cost: your model’s architecture is hidden
within the call() method, so Keras cannot easily inspect it, it cannot save or clone it,
and when you call the summary() method, you only get a list of layers, without any
information on how they are connected to each other. Moreover, Keras cannot check
types and shapes ahead of time, and it is easier to make mistakes. So unless you really
need that extra flexibility, you should probably stick to the Sequential API or the
Functional API.
'''



'''
You will typically have a script that trains a model and saves it, and one or more
scripts (or web services) that load the model and use it to make predictions. 
Loading the model is just as easy using the load model. 
However, this work only when using the Sequential API or the Functional API, 
but unfortunately not when using Model subclassing. 
However, you can use save_weights() and load_weights() to at leastsave and 
restore the model parameters (but you will need to save and restore everything else yourself).
'''
model.save("my_keras_model.h5")
model = keras.models.load_model("my_keras_model.h5")
model.save_weights("my_keras_weights.ckpt")
model.load_weights("my_keras_weights.ckpt")

'''
Checkpoint
-----------
'''
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
'''
when training on large datasets you should not only save your model at the end of
training, but also save checkpoints at regular intervals during training.
The fit() method accepts a callbacks argument that lets you specify a list of objects
that Keras will call during training at the start and end of training, at the start and end
of each epoch and even before and after processing each batch. For example, the 
ModelCheckpoint callback saves checkpoints of your model at regular intervals during
training, by default at the end of each epoch.

Moreover, if you use a validation set during training, you can set
save_best_only=True when creating the ModelCheckpoint. In this case, it will only
save your model when its performance on the validation set is the best so far. This
way, you do not need to worry about training for too long and overfitting the training
set: simply restore the last model saved after training, and this will be the best model
on the validation set. This is a simple way to implement early stopping (introduced in
Chapter 4):
'''
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])
    
    

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit(X_train, y_train, 
                    epochs=10,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb])
model = keras.models.load_model("my_keras_model.h5") # rollback to best model
mse_test = model.evaluate(X_test, y_test)


'''
Another way to implement early stopping is to simply use the EarlyStopping callback.
It will interrupt training when it measures no progress on the validation set for
a number of epochs (defined by the patience argument), and it will optionally roll
back to the best model. You can combine both callbacks to both save checkpoints of
your model (in case your computer crashes), and actually interrupt training early
when there is no more progress (to avoid wasting time and resources):
'''

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,  #interupt training when there is no progress on validaton set for 10 epochs
                                                  restore_best_weights=True)

'''
The number of epochs can be set to a large value since training will stop automatically
when there is no more progress. Moreover, there is no need to restore the best
model saved in this case since the EarlyStopping callback will keep track of the best
weights and restore them for us at the end of training.
There are many other callbacks available in the keras.callbacks
package. See https://keras.io/callbacks/.
'''
history = model.fit(X_train, y_train, 
                    epochs=100,     #can be set to a large value since training will stop automatically when there is no more progress
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
mse_test = model.evaluate(X_test, y_test)

'''
If you need extra control, you can easily write your own custom callbacks. 
For example, the following custom callback will display the ratio between the 
validation loss and the training loss during training (e.g., to detect overfitting):
'''


class PrintValTrainRatioCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


val_train_ratio_cb = PrintValTrainRatioCallback()
history = model.fit(X_train, y_train, 
                    epochs=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[val_train_ratio_cb])


'''
Visualization Using TensorBoard
-------------------------------
'''
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

'''
TensorBoard is a great interactive visualization tool that you can use to view the
learning curves during training, compare learning curves between multiple runs, visualize
the computation graph, analyze training statistics, view images generated by
your model, visualize complex multidimensional data projected down to 3D and
automatically clustered for you, and more! This tool is installed automatically when
you install TensorFlow, so you already have it!
'''
import os
root_logdir = os.path.join(os.curdir, "my_logs") #root log directory we will use for our TensorBoard logs

# small function that will generate a subdirectory path based on the current date
# and time, so that it is different at every run.
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
print('run log directory: ', run_logdir)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])    
model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

'''
Keras provide a nice tensorboard callback:   keras.callbacks.TensorBoard(run_logdir).
The TensorBoard callback take care of creating the log directory for you (along with
its parent directories if needed), and during training it will create event files and write
summaries to them. Even file are special binary log file containing the data you want to visualize.
The TensorBoard server will monitor the log directory, and it will automatically
pick up the changes and update the visualizations: this allows you to visualize
live data (with a short delay), such as the learning curves during training. 
In general, you want to point the TensorBoard server to a root log directory, and configure your
program so that it writes to a different subdirectory every time it runs. This way, the
same TensorBoard server instance will allow you to visualize and compare data from
multiple runs of your program, without getting everything mixed up.
'''
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])


#conda activate tf2
#cd ... (go to log file directory)
#tensorboard --logdir=./my_logs --port=6006
#http://localhost:6006

'''
Hyperparameter Tuning
---------------------
'''
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

'''
The flexibility of neural networks is also one of their main drawbacks: there are many
hyperparameters to tweak. Not only can you use any imaginable network architecture,
but even in a simple MLP you can change the number of layers, the number of
neurons per layer, the type of activation function to use in each layer, the weight 
initialization logic, and much more. How do you know what combination of hyperparameters
is the best for your task?

One option is to simply try many combinations of hyperparameters and see which
one works best on the validation set (or using K-fold cross-validation). 
For example, we can use GridSearchCV or RandomizedSearchCV to explore the hyperparameter
space, as we did in Chapter 2

To do this, we need to wrap our Keras models in objects that mimic regular Scikit-Learn 
regressors. The first step is to create a function that will build and compile a Keras model, 
given a set of hyperparameters.
'''

# build and compile a Keras model, given a set of hyperparameters
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model

 
'''
Next, we create a KerasRegressor based on this build_model() function.
The KerasRegressor object is a thin wrapper around the Keras model built using
build_model(). Since we did not specify any hyperparameter when creating it, it will
just use the default hyperparameters we defined in build_model(). Now we can use
this object like a regular Scikit-Learn regressor: we can train it using its fit()
method, then evaluate it using its score() method, and use it to make predictions
using its predict() method. Note that any extra parameter you pass to the fit()
method will simply get passed to the underlying Keras model. Also note that the
score will be the opposite of the MSE because Scikit-Learn wants scores, not losses
(i.e., higher should be better).
'''
keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

keras_reg.fit(X_train, y_train, epochs=100,
              validation_data=(X_valid, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])

mse_test = keras_reg.score(X_test, y_test)
X_new = X_test[:3]
y_pred = keras_reg.predict(X_new)

'''
However, we do not actually want to train and evaluate a single model like this, we
want to train hundreds of variants and see which one performs best on the validation
set. Since there are many hyperparameters, it is preferable to use a randomized search
rather than grid search (as we discussed in Chapter 2). Let’s try to explore the number
of hidden layers, the number of neurons and the learning rate
'''
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

# Fitting 3 folds for each of 10 candidates, totalling 30 fits
# Note that RandomizedSearchCV uses K-fold cross-validation, so it
# does not use X_valid and y_valid. These are just used for early stopping
rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)]) #we pass extra params to fit(): they simply get relayed to the 
                                                                          #underlying Keras models.

'''
The exploration may last many hours depending on the hardware, the size of the
dataset, the complexity of the model and the value of n_iter and cv. When it is over,
you can access the best parameters found, the best score, and the trained Keras model
'''
print('Best Params: ',rnd_search_cv.best_params_)
print('Best score: ',rnd_search_cv.best_score_)
print('Best estimator: ',rnd_search_cv.best_estimator_)
rnd_search_cv.score(X_test, y_test)
model = rnd_search_cv.best_estimator_.model
print(model)
model.evaluate(X_test, y_test)

'''
You can now save this model, evaluate it on the test set, and if you are satisfied with
its performance, deploy it to production. Using randomized search is not too hard,
and it works well for many fairly simple problems. However, when training is slow
(e.g., for more complex problems with larger datasets), this approach will only
explore a tiny portion of the hyperparameter space. You can partially alleviate this
problem by assisting the search process manually: first run a quick random search
using wide ranges of hyperparameter values, then run another search using smaller
ranges of values centered on the best ones found during the first run, and so on. This
will hopefully zoom in to a good set of hyperparameters. However, this is very time
consuming, and probably not the best use of your time.
Fortunately, there are many techniques to explore a search space much more efficiently
than randomly. Their core idea is simple: when a region of the space turns out 
to be good, it should be explored more. This takes care of the “zooming” process for
you and leads to much better solutions in much less time.
'''


'''
Number of Hidden Layer
----------------------
For many problems, you can just begin with a single hidden layer and you will get
reasonable results. It has actually been shown that an MLP with just one hidden layer
can model even the most complex functions provided it has enough neurons. For a
long time, these facts convinced researchers that there was no need to investigate any
deeper neural networks. But they overlooked the fact that deep networks have a much
higher parameter efficiency than shallow ones: they can model complex functions
using exponentially fewer neurons than shallow nets, allowing them to reach much
better performance with the same amount of training data.

This is due to the fact that Realworld data is often structured in such a hierarchical way 
and Deep Neural Networks automatically take advantage of this fact: lower hidden layers model 
low-level structures (e.g., line segments of various shapes and orientations), intermediate hidden
layers combine these low-level structures to model intermediate-level structures (e.g.,
squares, circles), and the highest hidden layers and the output layer combine these
intermediate structures to model high-level structures (e.g., faces)

Not only does this hierarchical architecture help DNNs converge faster to a good solution,
it also improves their ability to generalize to new datasets. For example, if you
have already trained a model to recognize faces in pictures, and you now want to
train a new neural network to recognize hairstyles, then you can kickstart training by
reusing the lower layers of the first network. Instead of randomly initializing the
weights and biases of the first few layers of the new neural network, you can initialize
them to the value of the weights and biases of the lower layers of the first network.
This way the network will not have to learn from scratch all the low-level structures
that occur in most pictures; it will only have to learn the higher-level structures (e.g.,
hairstyles). This is called transfer learning.
In summary, for many problems you can start with just one or two hidden layers and
it will work just fine (e.g., you can easily reach above 97% accuracy on the MNIST
dataset using just one hidden layer with a few hundred neurons, and above 98% accuracy
using two hidden layers with the same total amount of neurons, in roughly the
same amount of training time). For more complex problems, you can gradually ramp
up the number of hidden layers, until you start overfitting the training set. Very complex
tasks, such as large image classification or speech recognition, typically require
networks with dozens of layers (or even hundreds, but not fully connected ones, as
we will see in Chapter 14), and they need a huge amount of training data. However,
you will rarely have to train such networks from scratch: it is much more common to
reuse parts of a pretrained state-of-the-art network that performs a similar task.
Training will be a lot faster and require much less data
'''


'''
Number of Neurons per Hidden Layer
----------------------------------
As for the hidden layers, it used to be a common practice to size them to form a pyramid,
with fewer and fewer neurons at each layer—the rationale being that many lowlevel
features can coalesce into far fewer high-level features. For example, a typical
neural network for MNIST may have three hidden layers, the first with 300 neurons,
the second with 200, and the third with 100. However, this practice has been largely
abandoned now, as it seems that simply using the same number of neurons in all hidden
layers performs just as well in most cases, or even better, and there is just one
hyperparameter to tune instead of one per layer—for example, all hidden layers could
simply have 150 neurons. However, depending on the dataset, it can sometimes help
to make the first hidden layer bigger than the others.

Instead of looking for the right hyper params, a simpler approach is to pick a model with more layers 
and neurons than you actually need, then use early stopping to prevent it from overfitting 
(and other regularization techniques, such as dropout, as we will see in Chapter 11). This has been
dubbed the “stretch pants” approach:17 instead of wasting time looking for pants that
perfectly match your size, just use large stretch pants that will shrink down to the
right size.
'''

'''
Hpyer params tuning example
--------------------------
Train a deep MLP on the MNIST dataset (you can load it using keras.datasets.mnist.load_data(). 
See if you can get over 98% precision. Try searching for the optimal learning rate by using 
the approach presented in this chapter (i.e., by growing the learning rate exponentially, 
plotting the loss, and finding the point where the loss shoots up). Try adding all the bells 
and whistles—save checkpoints, use early stopping, and plot learning curves using TensorBoard.
'''
import tensorflow as tf
from tensorflow.keras.datasets import mnist
(X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
print(X_train_full.shape)

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

plt.imshow(X_train[0], cmap="binary")
plt.axis('off')
plt.show()

K = tf.keras.backend

# Thi is a callback. Remember that callback are call during training: 
#   - at the start and end of training, 
#   - at the start and end of each epoch 
#   - and even before and after processing each batch
class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)
        #the last line set the learning rate to a by factor increased value
        #this is the same as lr = lr * factor
        #the method on_batch_end is inherited from the Callback class
        
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

#We will start with a small learning rate of 1e-3, and grow it by 0.5% at each iteration:
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])
expon_lr = ExponentialLearningRate(factor=1.005)

#Now let's train the model for just 1 epoch:
history = model.fit(X_train, y_train, epochs=1,
                    validation_data=(X_valid, y_valid),
                    callbacks=[expon_lr])   
                    # the callback will be executed at the end of each batch
                    #since we didn't specify a batch_size, the default (32) is used
                    # this mean they will be X_train.shape[0]/32 = 1719 run
                    # for which we will have the lr and the losses, as we store
                    # them our callback class

# We can now plot the loss as a functionof the learning rate:
plt.plot(expon_lr.rates, expon_lr.losses)
plt.gca().set_xscale('log')
plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
plt.xlabel("Learning rate")
plt.ylabel("Loss")

#the loss is low between 1e-1 ans 3e-1
#The loss starts shooting back up violently around 3e-1, 
#so let's try using 2e-1 as our learning rate:
tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=2e-1),
              metrics=["accuracy"])

run_index = 1 # increment this at every run
run_logdir = os.path.join(os.curdir, "my_mnist_logs", "run_{:03d}".format(run_index))
run_logdir

'./my_mnist_logs/run_001'

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_mnist_model.h5", save_best_only=True)
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb])

model = keras.models.load_model("my_mnist_model.h5") # rollback to best model
model.evaluate(X_test, y_test)


#We got over 98% accuracy. Finally, let's look at the learning curves using TensorBoard:
#%tensorboard --logdir=./my_mnist_logs --port=6006

