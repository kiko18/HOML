# https://lime-ml.readthedocs.io/en/latest/
'''
Intuition
---------
Because we want to be model-agnostic, what we can do to learn the behavior of the 
underlying model is to perturb the input and see how the predictions change. 
This turns out to be a benefit in terms of interpretability, because we can perturb 
the input by changing components that make sense to humans (e.g., words or parts of an image), 
even if the model is using much more complicated components as features (e.g., word embeddings).

The key intuition behind LIME is that it is much easier to approximate a black-box model by 
a simple model locally (in the neighborhood of the prediction we want to explain), as opposed 
to trying to approximate a model globally.

Imagine we want to explain a classifier that predicts how likely it is for the image 
to contain a tree frog. We take the image on the left and divide it into interpretable 
components (contiguous superpixels).

we then generate a data set of perturbed instances by turning some of the interpretable 
components “off” (in this case, making them gray). For each perturbed instance, 
we get the probability that a tree frog is in the image according to the model. 

We then learn a simple (linear) model on this data set, which is locally weighted —that is, 
we care more about making mistakes in perturbed instances that are more similar to the original image. 
In the end, we present the superpixels with highest positive weights as an explanation, 
graying out everything else.
'''

import os,sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import inception_v3 as inc_net
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
#from skimage.io import imread
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np

from skimage.segmentation import mark_boundaries

import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image

# Here is a simpler example of the use of LIME for image classification by using Keras (v2 or greater)
print('Notebook run using keras:', keras.__version__)

# Here we create a standard InceptionV3 pretrained model and use it on images by 
#first preprocessing them with the preprocessing tools
inet_model = inc_net.InceptionV3()

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)

# Let's see the top 5 prediction for some image
images = transform_img_fn([os.path.join('data','cat_mouse.jpg')])
# I'm dividing by 2 and adding 0.5 because of how this Inception represents images
plt.imshow(images[0] / 2 + 0.5)
plt.show()
preds = inet_model.predict(images)
for x in decode_predictions(preds)[0]:
    print(x)


#Explanation
#Now let's get an explanation
explainer = lime_image.LimeImageExplainer()

# hide_color is the color for a superpixel turned OFF. 
# Alternatively, if it is NONE, the superpixel will be replaced by the average of its pixels.
# Here, we set it to 0 (in the representation used by inception model, 0 means gray)
# top_labels: produce explanations for the K labels with highest prediction probabilities.
# If None, produce explanations for all K labels.
explanation = explainer.explain_instance(images[0].astype('double'), inet_model.predict, top_labels=5, hide_color=0, num_samples=1000)

# Now let's see the explanation for the top class
# We can see the top 5 superpixels that are most positive towards the class with the rest of the image hidden
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()