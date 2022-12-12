# Lung Nodule Detection in Chest X-RAY Images using Machine Learning and Deep learning approaches.

JSRT Dataset- https://www.kaggle.com/datasets/raddar/nodules-in-chest-xrays-jsrt
Kaggle Dataset - https://drive.google.com/drive/folders/130SQPuNvnrYYDZa-CYzugBmIQykoUAZV?usp=share_link

### About the Dataset:
In this dataset there are total of 247 X-ray images, in which males and females X-ray images are available. The ratio of males and females are as follows: Females - 52% and Males - 48% . Regular chest X-Ray images without any diseases are 93 X-ray images in total, from JPCNN001 to JPCNN093. For the uneven X-Ray scans images with lung nodules and its various types 154 X-ray images in total, from JPCLN001 to JPCLN154. 

### Proposed work 
> **SVM Approach:** 
We have started with the SVM approach for classification of Lung Nodules. We’ve trained the SVM model for 80% and tested the dataset for 20%.  We’ve also found the best parameters for this. Best parameters for SVM are: {'C': 0.1, 'gamma': 1, 'kernel': 'linear'}. The accuracy is 50%. At the end we’ve generated a confusion matrix and plotted the RP curve.

> KNN Approach:
We have then implemented the KNN algorithm for classification of Lung Nodules. We’ve trained the KNN model for 80% and tested the dataset for 20%. The steps involved in this are Selecting the number K of the neighbors, Calculating the Euclidean distance of K number of neighbors. Taking the K nearest neighbors as per the calculated Euclidean distance. Among these k neighbors, count the number of the data points in each category. Assigning the new data points to that category for which the number of the neighbor is maximum. At the end we performed k fold cross validation for KNN. The results are as follows: Cross-Validation accuracy is 
0.600 +/- 0.101. At the end we’ve generated a confusion matrix and plotted the RP curve.

> CNN Approach:
We have then implemented a CNN model. The architecture consists of 3 layers. In each layer there is a 2d-convolution layer followed by a max-pooling layer. At the end we have used softmax-layer for classification. Adam Optimizer is used and for loss function binary_crossentropy is used. We’ve used Denser Layer 1 with 256 nodes and Denser Layer 2 with 2 nodes using the softmax classification layer. The results are as follows 
loss: 0.4537 - accuracy: 0.6232.  At the end we’ve generated a confusion matrix.

> MLP Approach:
We’ve then implemented a MLP model for classification. In MPL we’ve used parameters like activation, logistic, Learning rate, hidden layer size and maximum iterations. In activation parameters, we’ve used logistic, tanh and relu. For Learning rate we’ve used constant and adaptive.For hidden layer size, we’ve used the size of (64,32,16,8),(128,64,32,16,8) and (32,16,8). And lastly the maximum iterations we’ve used are 1000, 1500 and 2000. We’ve used Gridsearch as a classifier. Best parameters for MLP:  {'activation': 'logistic', 'hidden_layer_sizes': (64, 32, 16, 8), 'learning_rate': 'constant', 'max_iter': 2000}. At the end we’ve generated a confusion matrix and plotted the RP curve.

> Scale-Space Blob Detection using Increase Filter Method:
We’ve then implemented this Scale-Space Blob Detection using the Increase filter method for detecting the lung nodule at a particular position. The steps followed here are Getting the squared Laplacian response in scale-space., Ensuring the odd filter size, Initializing filter matrix and index of center entry, Obtaining a filter with normalization and applied  convolution and Updating the sigma value. The non-max suppression will first select the bounding box with the highest objectiveness score. And then remove all the other boxes with high overlap. To compute the radius we’ve used the helper function get_radius, this will compute the radius of each local maximum. 

> Scale-Space Blob Detection using Down Sample Method:
Finally, We’ve then implemented this Scale-Space Blob Detection using the Down Sample Method for detecting the lung nodule at a particular position. The steps involved in this are Ensuring odd filter size, Initializing filter matrix, Obtaining filter (no normalization needed). Scaling the image- Down scale, Applying convolution without normalization, Upscaling the image and Updating the sigma value. We’ve used two methods to set the pixels’ values of a maxima pixel to zero, they are rank filter and generic filter method. A matrix mask of certain dimension (must be equal to or greater than (3x3) is used with this function and non-maxima pixels are set to 0 on every scale of the scale-space array. 

### How to Run our Project
>  Unzip this folder and run the each cell for the files B22PS01_BTP_Final and Best_Parameters_B22PS01 (This is for the Classfication and Detection using JSRT dataset).

>  If you wanna run for kaggle dataset then download the dataset from starting mentioned link and hit 
"python SVM_KNN_CNN_MidReview.py" 

> After this user interface will be appeared then in 'upload lung cancer dataset' button, upload features folder. For traing and splitting the data we've used 80:20 %, later on excute SVM, KNN and CNN for the classification. Hence you can predict the output.

