# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 21:33:40 2020

@author: BT
"""


'''
The task of classifying and localizing multiple objects in an image is called object
detection. Until a few years ago, a common approach was to take a CNN that was
trained to classify and locate a single object, then slide it across the image, hence
the name sliding window. Moreover, since objects can have varying sizes, you would also 
slide the CNN across regions of different sizes.

This technique is fairly straightforward, but as you can see it will detect the same
object multiple times, at SLIGHTLY different positions. Some post-processing will then
be needed to get rid of all the unnecessary bounding boxes. A common approach for
this is called non-max suppression.

Here’s how you do it:
1-  First, you need to add an extra objectness output to your CNN, to estimate the
    probability that a flower is indeed present in the image (alternatively, you could
    add a “no-flower” class, but this usually does not work as well). It must use the
    sigmoid activation function, and you can train it using binary cross-entropy loss.
    Then get rid of all the bounding boxes for which the objectness score is below some 
    threshold: this will drop all the bounding boxes that don’t actually contain a flower.
    
2-  Find the bounding box with the highest objectness score and get rid of all the
    other bounding boxes that overlap a lot with it (e.g., with an IoU greater than 60%).

3-  Repeat step two until there are no more bounding boxes to get rid of.
'''

'''
Sliding window works pretty well, but it requires running the CNN many times 
(slide it across the entire image + slide it for different window size), 
so it is quite slow. Fortunately, there is a much faster way to slide a CNN across 
an image: using a fully convolutional network (FCN).
'''


'''
Fully Convolutional Networks (FCN)
---------------------------------
The idea of FCNs was first introduced in a 2015 paper25 by Jonathan Long et al., for
semantic segmentation (the task of classifying every pixel in an image according to
the class of the object it belongs to). The authors pointed out that you could replace
the dense layers at the top of a CNN by convolutional layers. To understand this, let’s
look at an example: suppose a dense layer with 200 neurons sits on top of a convolutional
layer that outputs 100 feature maps, each of size 7 × 7 (this is the feature map
size, not the kernel size). Each neuron will compute a weighted sum of all 100 × 7 × 7
activations from the convolutional layer (plus a bias term). Now let’s see what happens
if we replace the dense layer with a convolutional layer using 200 filters, each of
size 7 × 7, and with "valid" padding. This layer will output 200 feature maps, each 1
× 1 (since the kernel is exactly the size of the input feature maps and we are using
"valid" padding). In other words, it will output 200 numbers, just like the dense
layer did; and if you look closely at the computations performed by a convolutional
layer, you will notice that these numbers will be precisely the same as those the dense
layer produced. The only difference is that the dense layer’s output was a tensor of
shape [batch size, 200], while the convolutional layer will output a tensor of shape
[batch size, 1, 1, 200].

(To convert a dense layer to a convolutional layer, the number of filters
in the convolutional layer must be equal to the number of units
in the dense layer, the filter size must be equal to the size of the
input feature maps, and you must use "valid" padding. The stride
may be set to 1 or more, as we will see shortly.)

Why is FCN important? Well, while a dense layer expects a specific input size (since it
has one weight per input feature), a convolutional layer will happily process images of
any size26 (however, it does expect its inputs to have a specific number of channels,)
since each kernel contains a different set of weights for each input channel). Since an
FCN contains only convolutional layers (and pooling layers, which have the same
property), it can be trained and executed on images of any size!

For example, suppose we’d already trained a CNN for flower classification and localization.
It was trained on 224 × 224 images, and it outputs 10 numbers: outputs 0 to 4
are sent through the softmax activation function, and this gives the class probabilities
(one per class); output 5 is sent through the logistic activation function, and this gives
the objectness score; outputs 6 to 9 do not use any activation function, and they represent
the bounding box’s center coordinates, as well as its height and width. We can
now convert its dense layers to convolutional layers. In fact, we don’t even need to
retrain it; we can just copy the weights from the dense layers to the convolutional layers!
Alternatively, we could have converted the CNN into an FCN before training.

Now suppose the last convolutional layer before the output layer (also called the bottleneck
layer) outputs 7 × 7 feature maps when the network is fed a 224 × 224 image
(see the left side of Figure 14-25). If we feed the FCN a 448 × 448 image (see the right
side of Figure 14-25), the bottleneck layer will now output 14 × 14 feature maps.27
Since the dense output layer was replaced by a convolutional layer using 10 filters of
size 7 × 7, with "valid" padding and stride 1, the output will be composed of 10 features
maps, each of size 8 × 8 (since 14 – 7 + 1 = 8). In other words, the FCN will
process the whole image only once, and it will output an 8 × 8 grid where each cell
contains 10 numbers (5 class probabilities, 1 objectness score, and 4 bounding box
coordinates). It’s exactly like taking the original CNN and sliding it across the image
using 8 steps per row and 8 steps per column. To visualize this, imagine chopping the
original image into a 14 × 14 grid, then sliding a 7 × 7 window across this grid; there
will be 8 × 8 = 64 possible locations for the window, hence 8 × 8 predictions. However,
the FCN approach is much more efficient, since the network only looks at the
image once. In fact, You Only Look Once (YOLO) is the name of a very popular object
detection architecture. 

Several YOLO implementations built using TensorFlow are available on GitHub. In
particular, check out Zihao Zang’s TensorFlow 2 implementation. Other object detection
models are available in the TensorFlow Models project, many with pretrained 
weights; and some have even been ported to TF Hub, such as SSD31 and Faster-
RCNN,32 which are both quite popular. SSD is also a “single shot” detection model,
similar to YOLO. Faster R-CNN is more complex: the image first goes through a
CNN, then the output is passed to a Region Proposal Network (RPN) that proposes
bounding boxes that are most likely to contain an object, and a classifier is run for
each bounding box, based on the cropped output of the CNN.

The choice of detection system depends on many factors: speed, accuracy, available
pretrained models, training time, complexity, etc. The papers contain tables of metrics,
but there is quite a lot of variability in the testing environments, and the technologies
evolve so fast that it is difficult to make a fair comparison that will be useful for
most people and remain valid for more than a few months.

So, we can locate objects by drawing bounding boxes around them. Great! But perhaps
you want to be a bit more precise. Let’s see how to go down to the pixel level.
(i.e semantic segmentation). Before doing so let's take a look at one of the most
famous metric in object detection.
'''

'''
Mean Average Precision (mAP)
---------------------------
A very common metric used in object detection tasks is the mean Average Precision
(mAP). “Mean Average” sounds a bit redundant, doesn’t it? To understand this metric,
let’s go back to two classification metrics we discussed in Chapter 3: precision and
recall. Remember the trade-off: the higher the recall, the lower the precision. You can
visualize this in a precision/recall curve (see Figure 3-5). To summarize this curve
into a single number, we could compute its area under the curve (AUC). But note that
the precision/recall curve may contain a few sections where precision actually goes up
when recall increases, especially at low recall values (you can see this at the top left of
Figure 3-5). This is one of the motivations for the mAP metric.

Suppose the classifier has 90% precision at 10% recall, but 96% precision at 20%
recall. There’s really no trade-off here: it simply makes more sense to use the classifier
at 20% recall rather than at 10% recall, as you will get both higher recall and higher
precision. So instead of looking at the precision at 10% recall, we should really be
looking at the maximum precision that the classifier can offer with at least 10% recall.
It would be 96%, not 90%. Therefore, one way to get a fair idea of the model’s performance
is to compute the maximum precision you can get with at least 0% recall, then
10% recall, 20%, and so on up to 100%, and then calculate the mean of these maximum
precisions. This is called the Average Precision (AP) metric. Now when there are
more than two classes, we can compute the AP for each class, and then compute the
mean AP (mAP). That’s it!

In an object detection system, there is an additional level of complexity: what if the
system detected the correct class, but at the wrong location (i.e., the bounding box is
completely off)? Surely we should not count this as a positive prediction. One
approach is to define an IOU threshold: for example, we may consider that a prediction
is correct only if the IOU is greater than, say, 0.5, and the predicted class is correct.
The corresponding mAP is generally noted mAP@0.5 (or mAP@50%, or
sometimes just AP50). In some competitions (such as the PASCAL VOC challenge),
this is what is done. In others (such as the COCO competition), the mAP is computed
for different IOU thresholds (0.50, 0.55, 0.60, …, 0.95), and the final metric is the
mean of all these mAPs (noted AP@[.50:.95] or AP@[.50:0.05:.95]). Yes, that’s a mean
mean average.
'''

import numpy as np
import matplotlib.pyplot as plt

def maximum_precisions(precisions):
    return np.flip(np.maximum.accumulate(np.flip(precisions)))

recalls = np.linspace(0, 1, 11)

precisions = [0.91, 0.94, 0.96, 0.94, 0.95, 0.92, 0.80, 0.60, 0.45, 0.20, 0.10]
max_precisions = maximum_precisions(precisions)
mAP = max_precisions.mean()
plt.plot(recalls, precisions, "ro--", label="Precision")
plt.plot(recalls, max_precisions, "bo-", label="Max Precision")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.plot([0, 1], [mAP, mAP], "g:", linewidth=3, label="mAP")
plt.grid(True)
plt.axis([0, 1, 0, 1])
plt.legend(loc="lower center", fontsize=14)
plt.show()

