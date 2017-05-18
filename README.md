## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.


### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

### Dataset and Repository

1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.
2. Clone the project, which contains the Ipython notebook and the writeup template.
```sh
git clone https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project
cd CarND-Traffic-Sign-Classifier-Project
jupyter notebook Traffic_Sign_Classifier.ipynb
```

---

### Project Final Report

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/disp1.png "Visualization"
[image2]: ./examples/disp2.png "Grayscaling"
[image3]: ./examples/distr.png "Random Noise"
[image4]: ./examples/diste.png "Traffic Sign 1"
[image5]: ./examples/disva.png "Traffic Sign 2"
[image6]: ./examples/norm1.png "Traffic Sign 3"
[image7]: ./examples/norm2.png "Traffic Sign 4"
[image8]: ./examples/ex1.png "Traffic Sign 5"
[image9]: ./examples/ex2.png "Traffic Sign 5"
[image10]: ./examples/top1.png "Traffic Sign 5"
[image11]: ./examples/top2.png "Traffic Sign 5"
[image12]: ./examples/top3.png "Traffic Sign 5"
[image13]: ./examples/top4.png "Traffic Sign 5"
[image14]: ./examples/top5.png "Traffic Sign 5"
[image15]: ./examples/top6.png "Traffic Sign 5"
[image16]: ./examples/top7.png "Traffic Sign 5"
[image17]: ./examples/top8.png "Traffic Sign 5"
[image18]: ./examples/top9.png "Traffic Sign 5"
[image19]: ./examples/top10.png "Traffic Sign 5"

---
### README

here is a link to my [project code](https://github.com/RyannnG/CarND-Traffic-Signs/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the third code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the 4th-9th code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. I randomly choose a sample from the training set to how the dataset look like.

![alt text][image1]

![alt text][image2]

Then I use Pandas DataFrame plot method plotting bar charts of distribution of  each dataset's labels.

Here is the result:

![alt text][image3]

![alt text][image4]

![alt text][image5]

We can see that all train/test/valid dataset is not perfectly balanced, but they are consistant on  label distributions. 

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained **Pre-process the Data Set**  of the IPython notebook.

As a first step, I decided to convert the images to grayscale because LeCun's paper shows 

> we establish a new record of 99.17% accuracy by increasing our network’s capacity
> and depth and **ignoring color information**. This somewhat **contradicts prior results** with other methods suggesting that colorless recognition, while effective, was less accurate.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image6]

As a last step, I scale image data to (-0.5, 0.5),  because I'd like for each feature to have a similar range so that the gradients don't go out of control.

![alt text][image7]

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. 

I use the default data provided by the course link. The size of each dataset shows below

| Training   | 34799 |
| ---------- | ----- |
| Testing    | 12630 |
| Validation | 4410  |



If we want more data for validation, we could use `train_test_split()` from `sklearn.utils` to split trainging data.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the **Model Architecture** of the ipython notebook. 

My final model looked like LeNet and consisted of the following layers:

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |         32x32x1 grayscale image          |
| Convolution 5x5 | 1x1 stride, same padding, outputs 32x32x32 |
|      RELU       |                                          |
|   Max pooling   | 2x2 stride,  same padding,outputs 16x16x64 |
|     Dropout     |         0.8 (only for training)          |
| Convolution 5x5 | 1x1 stride, same padding, outputs 16x16x64 |
|      RELU       |                                          |
|   Max pooling   | 2x2 stride,  same padding,outputs 8x8x64 |
|     Dropout     |         0.8 (only for training)          |
| Convolution 5x5 | 1x1 stride, same padding, outputs 8x8x64 |
|      RELU       |                                          |
|   Max pooling   | 2x2 stride,  same padding,outputs 4x4x64 |
|     Dropout     |         0.6 (only for training)          |
| Fully connected |                1024 x 400                |
|     Dropout     |         0.6 (only for training)          |
| Fully connected |                 400 x 43                 |
|     Softmax     |                    43                    |
|                 |                                          |




#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the **Train && Save** of the ipython notebook. 

To train the model, I set batch size to 128 and iterated 20 epochs for getting a better result. I use Adam optimizer which works well in practice and compares favorably to other adaptive learning-method algorithms. The learning rate is set to 0.001with exponential decay to avoid overfitting.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of  97.7%
* validation set accuracy of  96.4%
* test set accuracy of  96.1%

I first use the LeNet architecture. The training loss drops very quickly but doesn't perform well on validation and test set. Then I change the depth of each convolution layer and add more hidden nodes in the fully connected layer. Then I also add dropout after each convolution layer and the first fully connected layer to avoid overfitting. The dropout rate on first two layer is 0.2 and 0.5 for the latter two layers.  The dropout layer shows astonishing  effects, the accuracy boost about 7% - 10%.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 German traffic signs that I found on the web:

![alt text][image8]

 After preprocess:

![alt text][image9]

 All these images are in high resolution. But after preprocess , they blurred. Take "Children crossing"( 9th image , label:28) for example, it's difficult to recognize whether there is a human or an animal  in the triangle section and may confused with Roadworks and Wild animals.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the **"Step 3"** of the Ipython notebook.

Here are the results of the prediction:

|                 Image                 |              Prediction               |
| :-----------------------------------: | :-----------------------------------: |
|           Speed limit 60km            |           Speed limit 60km            |
|     Dangerous curve to the right      |     Dangerous curve to the right      |
|               No entry                |               No entry                |
|               road work               |               road work               |
|                 stop                  |                 stop                  |
| Right of way at the next intersection | Right of way at the next intersection |
|              keep right               |              keep right               |
|             Traffic signs             |             Traffic signs             |
|           Children Crossing           |           Children Crossing           |
|                 Yield                 |                 Yield                 |

The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.1%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located at the end  of the Ipython notebook.

For all images, the model is sure that which sign is the image (probability greater than 0.7). The top five soft max probabilities were

![alt text][image10]

![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

![alt text][image16]

![alt text][image17]

![alt text][image18]



![alt text][image19]

