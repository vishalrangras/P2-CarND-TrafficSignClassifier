# **Traffic Sign Recognition** # 

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

[image1]: ./notebook_images/dataset-visualization.png "Visualization"
[image2]: ./notebook_images/preprocessing.png "Pre-processing"
[image3]: ./notebook_images/internet_images.png "Internet Images"
[image4]: ./notebook_images/010.jpg "Traffic Sign 1"
[image5]: ./notebook_images/001.jpg "Traffic Sign 2"
[image6]: ./notebook_images/011.jpg "Traffic Sign 3"
[image7]: ./notebook_images/008.jpg "Traffic Sign 4"
[image8]: ./notebook_images/009.jpg "Traffic Sign 5"
[image9]: ./notebook_images/003.jpg "Traffic Sign 6"
[image10]: ./notebook_images/006.jpg "Traffic Sign 7"
[image11]: ./notebook_images/005.jpg "Traffic Sign 8"
[image12]: ./notebook_images/004.jpg "Traffic Sign 9"
[image13]: ./notebook_images/002.jpg "Traffic Sign 10"
[image14]: ./notebook_images/LeNet.png "LeNet Model"

## Rubric Points ##
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README ###

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code. ####

You're reading it! and here is a link to my [project code](https://github.com/vishalrangras/Udacity-SDC-Projects/blob/master/P2-Traffic-Signal-Classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration ###

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually. ####

The code for this step is contained in the second code cell "In[3]" of the IPython notebook.  

I loaded the dataset using pickle load() function. Then training, validation and test datasets were stored in numpy arrays. The shapes of these arrays were shown using numpy shape attribute. Number of training, validation and test examples were shown using len function of python. To see the number of output classes, np.unique() function was used:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file. ####

The code for this step is contained in the third code cell "In[4]" of the IPython notebook.  

To visualize the data, I first printed the CSV file which gave me idea about which traffic sign each number represents in label dataset. Then I used matplotlib to plot a grid of 3 x 5 representing traffic signs and their corresponding labels on x-axis. Here is an exploratory visualization of the data set.

![alt text][image1]

### Design and Test a Model Architecture ###

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. ####

The code for this step is contained in the fifth and sixth code cell "In[6]" and In[9] of the IPython notebook.
In pre-processing section itself, I grayscaled images by passing image data to np.sum() function and dividing by 3. This in turn resulted into combining all the 3 channels of images into 1 channel. I could have also used cv2.imread with parameter 0 next to filename to load grayscale images but I wanted to try with a simply approach so I went with numpy method. During the lectures as well as on P2 slack channel, various methods for data standardization, normalization and augmentation were discussed but I wanted to keep it simple due to lack of time, so I just used grayscaling, shuffling and random seed as part of data pre-processing.

Here is an example of a traffic sign image before and after pre-processing.

![alt text][image2]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data) ####

The code for splitting the data into training and validation sets is contained in the fourth code cell "In[5]" of the IPython notebook.  

The image data which I took from the URL provided in lectures had validation set also contained in it. But using that validation set, I was not getting desired optimization of weight due to less randomness in the validation set so I skipped that validation set and created validation set from training set itself.

I split training data into training set and validation set using sklearn's train_test_split() function. I used random state, random seed and shuffling which helped in improving learning and validation accuracy. Without using random seed and random state, the validation accuracy was around 0.834. Using random seed improved it drastically to 0.892.

My final training set had 27839 number of images. My validation set and test set had 6960 and 12630 number of images.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model. ####

The code for my final model is located in the eighth and ninth cell "In[11]" and "In[12]" of the ipython notebook. 

## LeNet Model Architecture along with description for Traffic Signal classifier ##
![alt text][image14]

As it was explained in the Project Module by David, I have modified the LeNet model to accomodate 43 classes of Traffic Signal instead of 10 classes of MNIST data. There are no other significant changes in the following model. Here is the description for various layers of this Model which I have described based on my understanding of LeNet model:

### Input ##
The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since traffic signal images are converted into grayscale during pre-processing of data, C is 1 in this case.

### Architecture ###
**Layer 1: Convolutional.** Patch size is 5 x 5. Output depths is 6. Stride is of 1 x 1 with Valid padding. The output shape will be 28x28x6.

**Activation.** Relu as an activation.

**Pooling.** Used Max-pooling with a stride of 2 and Valid padding. The patch size is also 2 x 2. The output shape will be 14x14x6.

**Layer 2: Convolutional.** Patch size of 5 x 5. Stride of 1 x 1 with Valid padding and output depth of 16. The output shape will be 10x10x16.

**Activation.** Relu activation function.

**Pooling.** Max-pool with stride of 2, valid padding and patch size of 2 x 2. The output shape will be 5x5x16.

**Flatten.** Flattend the output shape of the final pooling layer such that it's 1D instead of 3D. Used `tf.contrib.layers.flatten` for this. Output will be 1D array of 5 x 5 x 16 = 400 values.

**Layer 3: Fully Connected.** Since we need output of this layer to have 120 outputs, we will provide a weight matrix of 400 (input) x 120 (output). Also, the bias matrix will be of 120 zeros. And then we do a matrix multiplication of Flatten layer with wieght matrix and add the bias matrix to it.

**Activation.** Relu activation function.

**Layer 4: Fully Connected.** This layer is similar to Layer 3 with only difference that it will yeild 84 outputs.

**Activation.** Relu activation.

**Layer 5: Fully Connected (Logits).** This layer is the layer which will emit Logits, which then will be hot-encoded and cross-entropy will be calculated for them for optimization of weights during training process. Since we have 43 different classes in our traffic signal data, the output of this fully connected layer should be 43. Hence we choose the weight and bias matrix accordingly.

### Output ###
Return the result of the 2nd fully connected layer in form of Logits. This Layer is Layer 5 in the above architecture.

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate. ####

The code for training the model is located in the tenth and eleventh cell "In[29]" and "In[30]" of the ipython notebook. 

My model used LeNet architecture as-is where I just modified output dimensions to predict for 43 classes instead of 10.
I used batch size of 128 as it was working fine and I didn't find the need to modify it. I trained the model on different epochs and learning rates which I have explained in below point #5. I used AdamOptimizer as-is which has the benefits of moving averages of parameters (momentum) and converges quickly without hyper-parameter tuning requirements. The learning rate was tried with different values as in this order: 0.001, 0.009, 0.007, 0.005, 0.003, 0.001. As I was getting desired results at 0.001 as well, I kept it to this rate for my final model evaluation. Higher learning rate of 0.009 was having little oscillations but it variation in accuracy was negligible in my case so it didn't bothered much. Ultimately, I reverted back to the best working one which was 0.001.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem. ####

The code for calculating the accuracy of the model is located in the twelfth cell "In[31]" of the Ipython notebook.

My final model results were:
* training set accuracy of 0.988
* validation set accuracy of 0.978 (Epoch 27) 
* test set accuracy of 0.885

My process was iterative but I didn't had to modify the LeNet architecture to get the results. My model was constant, I changed certain hyperparamaters and data processing techniques which resulted into better accuracy. Here is the description of how I iteratively improved the accuracy of model:

1. I first started without any kind of image pre-processing and without creating validation data from training data. Only pre-processing which I applied was to shuffle train data. I used the validation.p file as my validation data which was available in dataset zip file. As it turned out, my model started learning with validation accuracy of around 0.702 and improved to around 0.785. I tried this with 10 epochs, 15 epochs and then 25 epochs. In each iteration, model was not able to cross validation accuracy of 80%.

2. As next step, I split training data into 4:1 ratio of train:valid data. This lead to some validation accuracy 0.834 without random seed and 0.892 with random seed. My mentor suggested me to apply random seed to the shuffling process and it helped in improving the accuracy. I trained model with these conditions for 15 epochs only.

3. Next, I applied image pre-processing by simply converting image into grayscale. I mentioned above in the pre-process section the approach taken to grayscale the image using numpy. I am aware that there are more methods possible which can really make the model learn better such as data augmentation, creating fake data, normalization and standardization. Due to lack of time, I skipped those methods as I was able to get nice accuracy on my model with just above mentioned techniques.

When I tried this LeNet model on Internet images, it gave me 70% accuracy so little bit more tweaking on this model and making it learn from the bad data to predict can lead to even more robust traffic sign classifier as per my understanding.

I also tried to apply Modified Lenet Architecture for Traffic Signal Classifier based on the paper Traffic Sign Recognition with Multi-Scale Convolutional Networks but my implementation for that model resulted into a very low validation accuracy of 0.65 and as a result I had to drop it for the time being. In the upcoming time, I intend to work on that model more and improve accuracy, but I do not want to include it as a part of submission as of now. This is just an FYI.

### Test a Model on New Images ###

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify. ####

Here are ten German traffic signs that I found on the web:

![alt text][image4] 

![alt text][image5] 

![alt text][image6] 

![alt text][image7] 

![alt text][image8]

![alt text][image9] 

![alt text][image10] 

![alt text][image11] 

![alt text][image12] 

![alt text][image13]

The images are having different sizes, angle orientation and luminious intensity which should be the good enough candidate for the model to try and predict based on its learning. I believe as the images are of different dimensions, scaling them or cropping them will have impact on prediction. Also, the images which are not clear in terms of luminous intensity or having not direct camera angle should be difficult. There is one image of slippery road which is covered with snow. I believe, model should not be able to predict this image due to presence of snow.

After resizing and pre-processing the internet images, the internet images appears as below:


![alt text][image3]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric). ####

The code for making predictions on my final model is located in the sixteenth, seventeenth and eighteenth cell "In[48]", "In[49]", "In[52]" of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery Road      	| Slippery Road   								| 
| Turn Left ahead     	| Turn Left ahead 								|
| 50 km/h				| End of speed limit (80km/h)					|
| 50 km/h	      		| Speed limit (80km/h)					 		|
| Stop					| Speed limit (80km/h)      					|
| Road work   	 	  	| Road work   									| 
| Slippery Road   		| Slippery Road 								|
| 60km/h				| 60km/h										|
| 30km/h	      		| 30km/h					 					|
| General caution		| General caution      							|

Input Labels = [23, 34,  2,  2,  14, 25, 23,  3,  1, 18]

Predicted Labels = [23, 34,  6,  5,  5,  25, 23,  3,  1, 18]

The model was able to correctly guess 7 of the 10 traffic signs, which gives an accuracy of 70%. This accuracy is less compared to the test set accuracy of 0.885. The three images which model was not able to classify correctly were all having very similar characteristics that they were almost round in shape with red colored background and speed limit written in the middle or Stop sign.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts) ####

The code for making predictions on my final model is located in the nineteenth cell "In[53]" of the Ipython notebook.

Here are the values of probabilities and image id along with each image:

![alt text][image4] 

Probabilities = [ 0.40267757  0.15005061  0.14909063  0.14909063  0.14909063]

Predicted Labels = [23, 19, 11, 34, 21]

The model predicts 0.4 probability for Slippery road which is the correct prediction. The other close predictions are : Dangerous curve to the left, Right-of-way at the next intersection, Turn left ahead and Double curve.

![alt text][image5] 

Probabilities = [ 0.40460971  0.14884759  0.14884759  0.14884759  0.14884759]

Predicted Labels = [34, 23, 17, 36, 38]

The model predicts 0.4 probability of Turn left ahead which is the correct prediction. Other close predictions are: Slippery road, No entry, Go straight or right and Keep right.

![alt text][image6] 

Probabilities = [ 0.30889761  0.21843128  0.15756191  0.1575551   0.15755409]

Predicted Labels = [ 6, 41,  1,  2,  8]

The model predicts 0.3 probability of End of speed limit (80km/h). Model is predicting wrong and the correct answer is speed limit of (50km/h). I believe due to round red coloured sign board, the model is not able to make correct predictions. Data augmentation can improve this.

![alt text][image7] 

Probabilities = [ 0.40442264  0.14896369  0.14887123  0.1488712   0.14887121]

Predicted Labels = [ 5,  6, 36,  1,  4]
The model predicts 0.4 probability of Speed limit (80km/h), which is also again wrong because this image is also of speed limit of (50km/h).

![alt text][image8]

Probabilities = [ 0.39317906  0.15471518  0.15154389  0.15028842  0.1502735 ]

Predicted Labels = [ 5, 40,  4, 11,  1]
The model predicts 0.3 probability of Speed limit (80km/h), which is also again wrong because this image is of Stop sign.

![alt text][image9] 

Probabilities = [ 0.40460971  0.14884759  0.14884759  0.14884759  0.14884759]

Predicted Labels = [25, 31, 23, 37, 17]
0.4 probability prediction for Road work sign which is a correct prediction.

![alt text][image10]

Probabilities = [ 0.40460971  0.14884759  0.14884759  0.14884759  0.14884759]

Predicted Labels = [23, 29, 21, 19, 36]
0.4 probability prediction of Slippery road which is also correct prediction.

![alt text][image11] 

Probabilities = [ 0.40460438  0.14885083  0.1488483   0.14884827  0.14884825]

Predicted Labels = [ 3, 31,  1,  6,  5]
0.3 probability prediction for speed limit of 60 km/hr which is a correct prediction.

![alt text][image12] 

Probabilities = [ 0.40460971  0.14884759  0.14884759  0.14884759  0.14884759]

Predicted Labels = [ 1,  2,  5,  0,  3]
0.4 probability prediction of Speed Limit 30 km/hr which is a correct prediction.

![alt text][image13]

Probabilities = [ 0.40460971  0.14884759  0.14884759  0.14884759  0.14884759]

Predicted Labels = [18, 27, 26, 21,  0]
0.4 probability prediction of General caution which again is the correct prediction.