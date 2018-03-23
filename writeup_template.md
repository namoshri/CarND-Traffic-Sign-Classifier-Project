# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/download.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[imagea]: ./examples/1.png "Traffic Sign 1"
[imageb]: ./examples/11.png "Traffic Sign 2"
[imagec]: ./examples/12.png "Traffic Sign 3"
[imaged]: ./examples/3.png "Traffic Sign 4"
[imagee]: ./examples/8.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README


You're reading it! and here is a link to my [project code](https://github.com/namoshri/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration


I used the python and csv library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

As a first step, I decided to convert the images to grayscale because as RGB use doesn't signifies much.

As a last step, I normalized further image data because it makes calculation much faster and gives better accuracy




My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 GrayScale image  						| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3	    | 1x1 stride, same padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Fully connected		| Input = 800. Output = 564 					|
| RELU					|												|
| DropOut				| 	        									|
| Fully connected		|Input = 564. Output = 324.    	    			|
| RELU					|												|
| DropOut				|         										|
| Fully connected		| Input = 324. Output = 43.						|
| Softmax				|         										|
|						|												|
|						|												|
 


#### 3. Training of Model

To train the model, I used an default AdamOptimizer optimized. Num of epochs set to 50, batch size is 128 and learning rate 0.001

#### 4. Results on various data:

My final model results were:
Training Accuracy = 0.999
Valid Accuracy = 0.965
Test Accuracy = 0.934



I tried LeNet as as first architecture reference. It was earlier used for MNIST . With change in classification label 
was able to use classify traffice signals with 0.87 accuracy*


It has less classification and accuracy was less. Errors were observed in few images.

I did lot of experiments in increasing convolution layer, changing output dimensions, adding multiple FC laters. 
Adjusting stride and size for pooling etc.

In final network, dropout parameter adjusted. Increasing keep_prob from 0.5. to 0.7 or 0.8 shown better accuracy.


Increasing output dimensions for convolution layer and adding dropout later in fully connected layers helped to get better accuracy. 



### Test a Model on New Images


#### image selection: 

Here are five German traffic signs that I found on the web:

![alt text][imagea]
![alt text][imageb]
![alt text][imagec]
![alt text][imaged]
![alt text][imagee]

I found these images on website, all are distinct and signifies different meanings.
[Reference set](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset)
Images choosen such away that, they are from different categories. "No entry" and "Stop" images are
not bright enough, so little difficult because lighting condition for our model.




####  prediction and accuracy:

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No entry     			| NO entry 										|
| Ahead only			| Ahead only									|
| Road work	      		| Road work 					 				|
| Turn left ahead		| Turn Left ahead     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.
Accuracy is higher compared to test data and validation image data.


The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0      				| Stop sign   									| 
| 0.0     				| NO entry 										|
| 0.0					| Ahead only									|
| 0.0	  				| Road work 					 				|
| 0.0					| Turn Left ahead     							|

based on probabilities of each of 5 iamges, it is clear that predicted image top class probability and other 4 have far less probability.



