# **Traffic Sign Recognition**

## Writeup

### I use this markdown file as a writeup for my Traffic Sign Recognition project.

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

[image1]: ./examples/visualization.JPG "Visualization"
[image2]: ./examples/grayscale.JPG "Grayscaling"
[image3]: ./examples/random_noise.JPG "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./summary/training_set_Y_histogram.JPG "Training set Y Histogram"
[image10]: ./summary/image_rotation.JPG "Image Rotation"
[image11]: ./summary/Gaussian_Lighting_Noise.JPG "Gaussian_Lighting_Noise"
[image12]: ./summary/Trainingset_after_dataAugmentation.JPG "new training set after data augmentation"
[image13]: ./summary/grayscale.JPG "grayscale"
[image14]: ./New_Images/DoubleCurve.png "new image 1"
[image15]: ./New_Images/german-road-signs-slippery.png "new image 2"
[image16]: ./New_Images/pedestrian.png "new image 3"
[image17]: ./New_Images/roadwork.png "new image 4"
[image18]: ./New_Images/wildAnimalCrossing.png "new image 5"
[image19]: ./summary/training_accuracy.JPG "mode accuracy"
[image20]: ./summary/TestImagePrediction.JPG "test image prediction"
[image21]: ./summary/outputFeature.JPG "output feature"




## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and you can find my project code in the file - `Traffic_Sign_Classifier_XCai.ipynb`

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python to calculate summary statistics of the traffic
signs data set:

* The size of training set is: len(X_train_orig) = 34799
* The size of the validation set is: len(X_valid_orig) = 4410
* The size of test set is: len(X_test_orig) = 12630
* The shape of a traffic sign image is: (32, 32, 3)
* The number of unique classes/labels in the data set is: len(set(y_train_orig)) = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram chart showing the distribution of the training class. As we can see, some classes are under-represented. I will use several data augmentation methods to add more images for the under-represented training classes. It will be described in the following paragraph.

![alt text][image9]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As first step,  I decided to use data augmentation methods to generate more training sets for the under-represented training classes.
If the number of one class images is less than 800, I use image rotation (function of `rotate_images`) and gaussian noise (function of `add_gaussian_noise`) to add more images for those under-represented training classes.  

Below is one example of data augmentation by using image rotation (function of `rotate_images`). As you can see, I rotated the image by 5 degrees.

![alt text][image10]

Below is one example of data augmentation by using gaussian lighting noise (function of `add_gaussian_noise`).

![alt text][image11]

There are other ways of data augmentation. Due to the limited time for the project, I only used the above two methods for the under-represented training classes. About 10% of the under-represented images used image rotation function and the rest of them used gaussian lighting noise augmentation.   

Below are comparison graphs before and after data augmentation. As you can see, the right graph shows the new distribution of training classes after applying the above mentioned data augmentation methods.

![alt text][image12]


Second step, I decided to convert the RGB images to grayscale (function of `grayscale_batch`).

Third step, I applied the normalization function (`image_normalize`) for the grayscaled images.  Here are examples of 5 traffic sign images before and after grayscaling/normalization.

![alt text][image13]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:------------------------------------------------:|
| Input         		| 32x32x1 grayscaled image   							|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    |  1x1 stride, same padding, outputs 10x10x16 |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		| 1x400      									|
| Fully connected	layer 1	with dropout| 1x120      |
| Fully connected	layer 2	with dropout| 1x84      |
| Softmax				| 43        									|




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, after several tryouts, I ended up using BATCH_SIZE = 64, EPOCHS = 30, AdamOptimizer with a learning rate = 0.01.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.968
* test set accuracy of 0.947

The model accuracy from different epochs is shown below:

![alt text][image19]

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture tried was the default LeNet without dropout and without data augmentation. I chose it because that model was what I learned from the course. I wanted to see what it would perform for the training and validation data sets.

* What were some problems with the initial architecture?

The problem was it ended up around less than 90% accuracy.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

After the initial architecture, I generated more datasets for the under-represented training classes by using image rotation and gaussian lighting noise. It ended up around 92% validation accuracy. Then I added dropout for the fully connected layers. It bumped the validation accuracy to 95%. I have a question here - when I added dropout to the CONV1 and CONV2 layers, it didn't help improving the validation accuracy. I'm a little bit surprised and would like to know why if reviewer has a answer to that. (Thanks in advance!)

* Which parameters were tuned? How were they adjusted and why?

I tuned the BATCH_SIZE. I tried 256, 128 and 64. I found 128 and 64 BATCH_SIZE having similar accuracy performance. The 256 BATCH_SIZE actually hurts the accuracy. I also increased the EPOCHS from 20 to 30. It helped a little bit on the accuracy.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

The important design choice I think is the data augmentation before applying the LeNet. Also adding pooling layers helps. Finally the dropout out layers add regularization and avoid overfitting.  

If a well known architecture was chosen:
* What architecture was chosen?

Here I chose the LetNet-5 architecture for the project. It has 2 convolution layers and 3 fully connected layers

* Why did you believe it would be relevant to the traffic sign application?

With more layers, it extracts more images features. With several try-outs, it works pretty good on the validation and test accuracy. Due to the time I have for the project, I haven't tried other architectures. I will definitely try other training architecture.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Data augmentation, normalization helped better training the model. Dropout helped avoiding the overfitting, thus helped the validation and testing accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image14] ![alt text][image15] ![alt text][image16] ![alt text][image17] ![alt text][image18]

All the 5 images are screenshots of _Road Traffic Signs in Germany_. The image qualities are very similar to the training datasets. It may be difficult to class for the high quality images because training datasets are not high quality images.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Double Curve      		| Double Curve   									|
| Slippery road     			| Slippery road 										|
| Pedestrians					| Pedestrians											|
| Road work	      		| Road work					 				|
| Wild animals crossing			| Wild animals crossing     					|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.6%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is in my Ipython notebook.

For all the five images, the model is with a probability of 1.0 predition accuracy.  Below is an example for the first image prediction. The top five soft max probabilities were

| Top 5 Probability for the 1st image         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 1.00         			| Double Curve   									|
| 0.00     				| Road work 										|
| 0.00					| Beware of ice/snow												|
| 0.00	      			| Road narrows on the right				 				|
| 0.00				    | Speed limit (30km/h)     							|


The prediction result for all the 5 test images is shown below:

![alt text][image20]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Below is the output feature from the CONV1 layer. For me, CONV1 layer captures the contrast in the sign's painted symbol.

![alt text][image21]
