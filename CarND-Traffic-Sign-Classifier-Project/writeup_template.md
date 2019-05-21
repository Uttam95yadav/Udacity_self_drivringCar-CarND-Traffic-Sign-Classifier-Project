# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data are plot in bar chart

```
hist, bins = np.histogram(y_train, bins=n_classes)
width =  0.8*(bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.show()
```

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it has three chennal Red Green Blue and it cr5eate complexity and having less info so we have to reduce the dimenision  (32,32,3) to (32,32,1)

Here is an example of a traffic sign image before and after grayscaling.
```
def preprocess(Image):
    #Convert to grayscale, e.g. single channel Y
    Image = 0.299 * Image[:, :, :, 0] + 0.587 * Image[:, :, :, 1] + 0.114 * Image[:, :, :, 2]

    #Scale features to be in [0, 1]
    Image = (Image / 255.).astype(np.float32)
    
    #adjust histogram
    for i in range(Image.shape[0]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Image[i] = exposure.equalize_adapthist(Image[i]) 
            
    return Image

def reshape(Img): # Add a single grayscale channel
    return Img.reshape(Img.shape + (1,))

```

![alt text][image2]
The images in the dataset are color images with 3 channels. At first the unprocessed image was
used for classification and the details will be discussed in the discussion and conclusion
section. The color channels just add computational complexity to the algorithm as the color
channels doesn’t carry any information about the shape of the sign, which is the main
distinctive feature for classification. So the image was converted to single channel. Then the
grayscale image is normalised to a scale of [0,1] instead of [0,255]. Apart from that, image
histogram equalisation is done for enhance the contrast of the image. The algorithm was
obtained from reference [1].
As a last step, I normalized the image data because ...
data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project.

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture essentially depends on LeNet 5 and the model discussed in class,
Convolutional Neural Network for TensorFlow. The model consists of two convolutional and
fully connected layers. The two convolutional layers are made using 'tf.nn.conv2d' function in
TensorFlow, with a stride of 1 and 'SAME' padding. Hence, there is no change in size for the
output. The biases are added using 'tf.nn.bias_add' function. The result of this layer is made
nonlinear by applying a relu activation function. Moreover, a Max-pooling operation, having
stride 2 and 'SAME" padding, is performed on the result of the relu activation function. Thus a
reduction of dimension from 32x32 to 16x16 will occur at the output of first convolutional
layer after maxpooling and reduction from 16x16 to 8x8 at the output of second layer.
The number of filters will increase the features extracted from the image. Thus in the first
convolutional layer we used a 64 layer filter, making the output dimension after maxpooling
16x16x64. On the second convolutional layer we use a 32 layer filter is used making the
resultant shape 8x8x32.
Now this matrix is flattened to form single dimensional tensor of size 2048, which is in turn
densely connected to a layer having 256 neurons. Moreover, this layer is connected with the
output layer having 43 output neurons. 


 ```
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 64), mean = 0, stddev = 0.1))
    conv1_b = tf.Variable(tf.zeros([64]))
    
    conv1 = tf.nn.conv2d(x, conv1_W, strides = [1, 1, 1, 1], padding = 'SAME')
    conv1 = tf.nn.bias_add(conv1, conv1_b)
    conv1=tf.nn.relu(conv1)
    conv1=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')


    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 32), mean = 0, stddev = 0.1))
    conv2_b = tf.Variable(tf.zeros([32]))
 

    conv2 = tf.nn.conv2d(conv1, conv2_W, strides = [1, 1, 1, 1], padding = 'SAME')
    conv2 = tf.nn.bias_add(conv2, conv2_b)
    conv2=tf.nn.relu(conv2)
    conv2=tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding = 'SAME')

    fc1   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W =tf.Variable(tf.truncated_normal(shape=(8*8*32, 256), mean = 0, stddev = 0.1))
    fc1_b = tf.Variable(tf.zeros([256]))
    fc1   = tf.matmul(fc1, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    fc1=tf.nn.dropout(fc1, keep_prob)
    logits_w = tf.Variable(tf.truncated_normal(shape=(256, 43), mean = 0, stddev = 0.1))
    logits_b = tf.Variable(tf.zeros([43]))
    logits=tf.add(tf.matmul(fc1, logits_w), logits_b)
    logits=tf.nn.dropout(logits, keep_prob)
    return logits
 ```
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
The parameters chosen are learning rate, number epochs, dropout probability, and the batch
size. Kernel size for filters and strides are given as constant values. The training pipeline is as
discussed below.
The tf.nn.softmax_cross_entropy_with_logits function is used to find the cross entropy
between logits (the predicted output) and the label using softmax function. tf.reduce_mean
function is used to find the mean of error, which is fed to an optimizer. Adam Optimizer
(tf.train.AdamOptimizer) implements an extension of stochastic gradient descent.
For evaluation we find the right predictions and find the mean of it to get the accuracy of
prediction. To find the accuracy during each epoch the process is repeated for every epoch.
To train the model, I used an 
rate = 0.0005
EPOCHS = 75     
BATCH_SIZE = 256

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of 0.973 
* test set accuracy of 0.963

I used the same LeNet architecture and preprocessing codes used in the  class. But
not able to get beyond 89%. Then to improve Added Drop Out function.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]
The obtained images are pre-processed prior to feeding into the prediction pipeline. The preprocessed stop sign image is shown in 

The first image might be difficult to classify because ...
The prediction performance matrix is shown in table , in which the
predicted class number is given in the first raw and actual label id is given in the later row. somrtime The
70 km sign was miss classified as 60 km sign the text written on the board is not clear and model confuse between 6 and 7

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
[ 4 12 13 14 17 18 25] <-predictions
[ 4 12 13 14 17 18 25] <-actual


The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

INFO:tensorflow:Restoring parameters from ./model
[[ 4 12 13 14 17 18 25]
 [ 0  9 12  8 22 26 29]
 [ 3 14 10  7 14 36  7]
 [20 25  9 13 13 24 30]
 [ 1 13 15  1  8 11 10]]
(top  5 predictions above) for each image

probability for top 5 predictions for each image:
0 [  8.43521237e-01   1.39000237e-01   1.71776041e-02   2.82055407e-04
   1.24577391e-05]
1 [  1.00000000e+00   1.61126456e-13   6.22789919e-14   1.63805533e-14
   1.24403591e-15]
2 [  1.00000000e+00   5.88796676e-12   1.00973741e-13   7.37455751e-14
   3.37061786e-14]
3 [  1.00000000e+00   3.58439478e-09   8.62970737e-11   3.33869460e-11
   1.33325399e-11]
4 [  9.99998927e-01   6.76217553e-07   1.02822398e-07   6.76087595e-08
   6.73772931e-08]
5 [  1.00000000e+00   6.17663752e-15   3.38936131e-16   7.00470241e-17
   1.82582270e-17]
6 [  9.21592951e-01   7.32222423e-02   3.10378149e-03   5.93403471e-04
   4.05217113e-04]
INFO:tensorflow:Restoring parameters from ./model
Test Accuracy = 1.000


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


