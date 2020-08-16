# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 00:57:47 2020

@author: BT
"""



'''
Memory Requirements
-------------------
Another problem with CNNs is that the convolutional layers require a huge amount of RAM. 
This is especially true during training, because the reverse pass of backpropagation
requires all the intermediate values computed during the forward pass.
For example, consider a convolutional layer with 5 × 5 filters, outputting 200 feature
maps of size 150 × 100, with stride 1 and SAME padding. If the input is a 150 × 100
RGB image (three channels), then the number of parameters is (5 × 5 × 3 + 1) × 200
= 15,200 (the +1 corresponds to the bias terms), which is fairly small compared to a
fully connected layer. (A fully connected layer with 150 × 100 neurons, each connected to all 
150 × 100 × 3 inputs, would have 1502 × 1002 × 3 = 675 million parameters!)

However, each of the 200 feature maps contains 150 × 100 neurons,
and each of these neurons needs to compute a weighted sum of its 5 × 5 × 3 = 75 inputs: 
that’s a total of 225 million float multiplications. Not as bad as a fully connected
layer, but still quite computationally intensive. Moreover, if the feature maps
are represented using 32-bit floats, then the convolutional layer’s output will occupy
200 × 150 × 100 × 32 = 96 million bits (12 MB) of RAM. (In the international system of units 
(SI), 1 MB = 1,000 kB = 1,000 × 1,000 bytes = 1,000 × 1,000 × 8 bits.)

And that’s just for one instance! If a training batch contains 100 instances, 
then this layer will use up 1.2 GB of RAM!
During inference (i.e., when making a prediction for a new instance) the RAM occupied
by one layer can be released as soon as the next layer has been computed, so you
only need as much RAM as required by two consecutive layers. But during training
everything computed during the forward pass needs to be preserved for the reverse
pass, so the amount of RAM needed is (at least) the total amount of RAM required by
all layers.

If training crashes because of an out-of-memory error, you can try
reducing the mini-batch size. Alternatively, you can try reducing
dimensionality using a stride, or removing a few layers. Or you can
try using 16-bit floats instead of 32-bit floats. Or you could distribute
the CNN across multiple devices.
'''