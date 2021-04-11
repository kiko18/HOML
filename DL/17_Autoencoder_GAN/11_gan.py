'''
Generative Adversarial Networks (GAN)
-------------------------------------
GAN were proposed in a 2014 paper by Ian Goodfellow et al., and although the idea got 
researchers excited almost instantly, it took a few years to overcome some of the 
difficulties of training GANs. Like many great ideas, it seems simple in hindsight: 
make NN compete against each other in the hope that this competition will push them 
to excel. 

As shown in Figure 17-15, a GAN is composed of two neural networks:
Generator:
  Takes a random distribution as input (typically Gaussian) + outputs some data, 
  (typically an image). You can think of the random inputs as the latent representations 
  (i.e., codings) of the image to be generated. So, as you can see, the generator offers 
  the same functionality as a decoder in a variational autoencoder, and it can be used 
  in the same way to generate new images (just feed it some Gaussian noise, and it outputs 
  a brand-new image). However, it is trained very differently, as we will soon see.
Discriminator
  Takes either a fake image from the generator or a real image from the training set
  as input, and must guess whether the input image is fake or real.

Training
--------
During training, the generator and the discriminator have opposite goals: 
the discriminator tries to tell fake images from real images, while the generator 
tries to produce images that look real enough to trick the discriminator. 

Because the GAN is composed of 2 networks with different objectives, it cannot be 
trained like a regular NN. Each training ITERATION is divided into 2 phases:

• In the first phase, we train the discriminator.
  A batch of real images is sampled from the training set and is completed with an 
  equal number of fake images produced by the generator. The labels are set to 0 
  for fake images and 1 for real images, and the discriminator is trained on this 
  labeled batch for one step, using the binary cross-entropy loss. 
  Importantly, backpropagation only optimizes the weights of the discriminator 
  during this phase.

• In the second phase, we train the generator. 
  We first use it to produce another batch of fake images, and once again the 
  discriminator is used to tell whether the images are fake or real. This time we 
  do not add real images in the batch, and all the labels are set to 1 (real): 
  in other words, we want the generator to produce images that the discriminator 
  will (wrongly) believe to be real! Crucially, the weights of the discriminator 
  are frozen during this step, so backpropagation only affects the weights of the 
  generator. 

To summarize:
1) Train the discriminator on fake (0) and real (1) images
2) Freeze Discriminator weight
3) Train generator to produice fake (1) real looking images.

The generator never actually sees any real images, yet it gradually learns to 
produce convincing fake images! All it gets is the gradients flowing back through 
the discriminator. 
Fortunately, the better the discriminator gets, the more information about the real 
images is contained in these secondhand gradients, so the generator can make 
significant progress.
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import utils

np.random.seed(42)
tf.random.set_seed(42)

# Build a simple GAN for Fashion MNIST
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

del X_train_full, y_train_full

# Size of coding layer (which is input of generator)
codings_size = 30

# The generator is similar to an autoencoder’s decoder
# Remember that it never see real images and learn to produce fake real looking images.
generator = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28]) #image produced
])

# The discriminator is a regular binary classifier (it takes an image as input and ends 
# with a Dense layer containing a single unit and using the sigmoid activation function). 
discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),#image as input
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")#fake or real?
])

# For the second phase of each training iteration, we also need the full GAN model 
# containing  the generator followed by the discriminator.
gan = keras.models.Sequential([generator, discriminator])

# Next, we need to compile these models. 
# As the discriminator is a binary classifier, we can naturally use the binary cross-entropy loss. 
# The generator will only be trained through the gan model, so we don't need to compile it at all. 
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

# Importantly, the discriminator should not be trained during the second phase, 
# so we make it non-trainable before compiling the gan model.
# Remember that the trainable attribute is taken into account by Keras only when compiling a model.
# So, in gan model, discriminator is not trainable, since gan is compiled after we frooze discriminator.
# On the other side, the discriminator is trainable if we call its fit() method or its 
# train_on_batch() method (which we will be using), since we frooze it after compiling it.
discriminator.trainable = False 

# The gan model is also a binary classifier, so it can use the binary cross-entropy loss. 
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

#Since the training loop is unusual, we cannot use the regular fit() method. 
# Instead, we will write a custom training loop. 
#For this, we first need to create a Dataset to iterate through the images.
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

# We are now ready to write the training loop. Let’s wrap it in a train_gan() function
def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        print("Epoch {}/{}".format(epoch + 1, n_epochs))              
        for X_batch in dataset: #[batch_size, 28, 28]

            # phase 1 - training the discriminator
            #In phase one we feed Gaussian noise to the generator to produce fake images,
            #and we complete this batch by concatenating an equal number of real images.
            #The targets y1 are set to 0 for fake images and 1 for real images. Then we train
            #the discriminator on this batch. Note that we set the discriminator’s trainable
            #attribute to True: this is only to get rid of a warning that Keras displays when it
            #notices that trainable is now False but was True when the model was compiled
            #(or vice versa).
            noise = tf.random.normal(shape=[batch_size, codings_size])          #generate gaussian noise
            generated_images = generator(noise)                                 #generate fake images
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)    #complete fake with real images
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)         #generate label for X_fake_and_real
            discriminator.trainable = True                                      #just to get rid of keras warning
            discriminator.train_on_batch(X_fake_and_real, y1)                   #train discriminator on this batch
            
            # phase 2 - training the generator
            # In phase 2, we feed the GAN some Gaussian noise.
            # GAN's generator will start by producing fake images,
            # then the discriminator will try to guess whether these images are fake or real.
            # We want the discriminator to believe that the fake images are real,
            # so the targets y2 are set to 1. Note that we set the trainable attribute to
            # False, once again to avoid a warning.
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)                               #we want discriminator to believe fake image are real
            discriminator.trainable = False                                     #just to get ride of warning
            gan.train_on_batch(noise, y2)                                       #feed gan some noise
            
        # That’s it! If you display the generated images, you will see that at
        # the end of the first epoch, they already start to look like (very noisy) 
        # Fashion MNIST images. Unfortunately, the images never really get much better 
        # than that, and you may even find epochs where the GAN seems to be forgetting 
        # what it learned. Why is that? Well, it turns out that training a GAN can be 
        # challenging. Let’s see why   
        utils.plot_multiple_images(generated_images, 8)                     
        plt.show()                                                    

train_gan(gan, dataset, batch_size, codings_size, n_epochs=5)