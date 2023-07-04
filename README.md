# Training-Artificial-Neural-Network
In this project, I performed experiments on artificial neural network (ANN) training and drew conclusions from the experimental results. I implemented and trained multi layer perceptron (MLP) and convolutional neural network (CNN) classifiers on CIFAR-10 dataset.

## Part2. Implementing a Convolutional Layer with NumPy

2.1. Convolutional Neural Networks (CNNs) are widely used at the center of deep learning algorithms. 
The previous networks, called traditional neural networks, are less effective than CNNs because of their 
input shape restrictions. In order to use these traditional neural networks, the input size should be a 
constant number. In addition, traditional neural networks have many parameters to be trained. As the 
number of parameters increases, the effectiveness of the network decreases. So, it is hard to train a 
traditional neural network.

On the other hand, CNNs have a better approach to these problems. It is not restricted to using
a fixed input shape while using CNNs. In other words, the shape of the input can be arbitrary. 
Furthermore, the number of parameters in the CNNs is decreased. That’s why it is easier to train a CNN 
when compared to a traditional neural network. 

Especially for computer vision and image processing tasks, CNNs are used a lot. The main 
reason is that CNNs can extract features from the input. These features are edges, corners, patterns, 
and textures. Object detection, segmentation, and image classification operations can be done 
effectively using these feature extraction abilities. To conclude, using CNNs in image processing tasks is
important since CNNs can extract features from inputs efficiently.

2.2. Kernels are filters that can extract features from the input image. A kernel is a matrix in which the 
weights are involved. Generally, the kernel size is chosen to be smaller than the input size in order to 
extract local features from the input. The kernel is convolved with the input image and gives an output 
smaller than the input image. The stride value specifies the movement of the kernel. For example, ifthe stride is selected as 1, then the kernel, or filter, moves one by one through the input matrix, and 
each movement kernel is performed a dot product operation with the input matrix. By doing so, highlevel features such as patterns, edges, and corners can be extracted from the input. 

The sizes of a kernel depend on the application. For example, a smaller kernel should be used 
to extract local features from the input. On the other hand, to extract larger features, a larger kernel 
can be used. The size of the kernel consists of two parameters, namely kernel height and kernel width. 
Kernel height is the number of rows in the kernel, and the kernel width is the number of columns in 
the kernel. Generally, kernels are chosen as a square matrix, but it is not mandatory. Rectangular 
kernels can be used as well.

2.3. After the convolutional layer, an output image is generated. This image has 5 rows and 8 columns. 
Each row corresponds to the same number class, and each column corresponds to a different kernel. 
The convolutional layer tried to extract features from the given input, and this resulted in 8 different 
channels for each number. But as it can be seen from the image, the pattern of the number is not 
understood yet to decide which number class is detected. That’s why the convolutional layer itself is 
not sufficient. According to different kernels, different outputs are obtained. It can be concluded that 
outputs of the same kernel looks like each other, and outputs of different kernels are different from 
each other.

<img src="./Results-Part2/hw1-part2-img.png">

2.4. The convolutional layer consists of eight 4x4 kernels. Each kernel has its own weights. So, after the 
convolution operation, output of the convolutional layer will give us 8 different images or matrices 
which are called channels. In the output image, rows correspond to different batches and columns 
corresponds to different kernels. When we examine the same column, almost all of the number in a 
column are similar to each other. The main reason of this similarity comes from the same kernel is 
applied to all of these numbers. Even though they are different numbers, the output channel looks likesimilar. In other words, kernel weights are important fact that affect the output of the convolutional layer. Output channels of the same kernel can look like to each other.

2.5. As it was stated in the previous question, the convolutional layer consists of eight 4x4 kernels, and 
each row corresponds to the same number class, but the output channels are not similar. The main 
reason of this issue is that different kernels have different weights, and the input weights are the same 
for the same row. So, for each convolution operation gives different output. For the same number, 
different kernels are applied. Even though the number does not change, the output channel looks 
different than the others. 

2.6. For feature extraction, using convolutional layers are important but using only one type of kernel 
will give us wrong intuitions about the input image. That’s why using more kernels with different weight 
could be better approach to extract patterns from the input images. Also, by using only one 
convolutional layer is not sufficient. Using more than one convolutional layer could give better results. 
Furthermore, after examining the output image, it can be understood that using only one convolutional 
layer cannot extract complex patterns and high-level features. This type of networks can be described 
as shallow networks and to make feature extraction of these networks better, deeper neural networks 
can be used. Deep neural networks can be created by using multiple convolutional layers that are 
stacked top on each other. In this way, more complex features can be extracted.

## Part3. Experimenting ANN Architectures 

### Case 1: Multi Layer Perceptron 1 (Mlp1)
<img src="./Results-Part3/mlp1/part3Plots.png">
<img src="./Results-Part3/mlp1/before_train_weights.png">
<img src="./Results-Part3/mlp1/input_weights.png">

### Case 2: Multi Layer Perceptron 2 (Mlp2)
<img src="./Results-Part3/mlp2/part3Plots.png">
<img src="./Results-Part3/mlp2/before_train_weights.png">
<img src="./Results-Part3/mlp2/input_weights.png">

### Case 3: Convolutional Neural Network 3 (CNN3)
<img src="./Results-Part3/cnn_3/part3Plots.png">
<img src="./Results-Part3/cnn_3/before_train_weights.png">
<img src="./Results-Part3/cnn_3/input_weights.png">

### Case 4: Convolutional Neural Network 4 (CNN4)
<img src="./Results-Part3/cnn_4/part3_plots.png">
<img src="./Results-Part3/cnn_4/before_train_weights.png">
<img src="./Results-Part3/cnn_4/input_weights.png">

### Case 5: Convolutional Neural Network 5 (CNN5)
<img src="./Results-Part3/cnn_5/part3_plots.png">
<img src="./Results-Part3/cnn_5/before_train_weights.png">
<img src="./Results-Part3/cnn_5/input_weights.png">

**Q1.** A classifier's generalization performance measures how effectively a classifier can correctly 
categorize new data that wasn't used during the training phase. In other words, it evaluates how well 
the classifier can apply the data it discovered from the training set to new, unexplored data.
A classifier with strong generalization performance may correctly categorize new data, even 
when different from the training data. On the other hand, the classifier may overfit the training data 
and perform poorly on new data if it has poor generalization performance.
Several metrics can be used to assess a classifier's generalization ability, including accuracy, 
precision, recall, F1-score, and area under the receiver operating characteristic curve (AUC-ROC). These 
metrics can be used to compare several classifiers' performance and ascertain how well the classifier 
works on new data.

**Q2.** The validation vs. training accuracy plot can give a better approach to inspect the 
generalization performance of the models since the training accuracy of the model most of the time 
increases, but the validation accuracy does not. The main reason for this issue is that the model starts 
to overfit the training data, especially for a high number of epochs and many runs that make the model 
overfit the data. That's why even though the model's training accuracy increases, the model's validation 
accuracy decreases. In addition to the validation accuracy vs. training accuracy plot, other types of plots 
can be used to inspect the model's generalization performance. These methods include Confusion 
Matrix, Prevision-Recall Curve, ROC Curve (Receiver Operating Characteristic), and Learning Curve. Still, 
in our case, we plotted the validation accuracy vs. training accuracy, training loss, and best test 
performance curves.

**Q3.** The generalization performance of the architectures can be compared by inspecting the 
validation accuracy vs. training accuracy plots. As we examine these plots, we can see that the 
multilayer perceptron architectures do not show a good generalization performance. Both mlp1 and 
mlp2 architectures have shown similar validation accuracy vs. training accuracy plots. In both cases, 
the training accuracy of the architectures increases. The training accuracies of the architectures started 
from 42 levels and rose to 52 levels.
On the other hand, the validation accuracies of the architectures remained at almost the same 
level, which is 38. Furthermore, as the training number increases, the validation accuracy of the 
architectures starts to decrease after a certain point which shows that these two architectures, mlp1, 
and mlp2 are not good at generalization. They overfit the data, and even though their training 
accuracies increased and their training losses decreased, the validation accuracies of these 
architectures show that they did not learn the pattern of the data correctly. 

Regarding the CNN architectures, cnn_3, cnn_4, and cnn_5 architectures showed similar 
performances in terms of generalization. When I examine the training losses of these CNN 
architectures, the training loss of these architectures tends to decrease in time. If we increase the 
number of runs, some of the architectures will give a better performance. In addition to the training 
losses, the validation accuracy vs. training accuracy plots shows that all of these CNN architectures are 
good at generalization. The cnn_3 architecture had 58 training accuracy and 54.5 validation accuracy 
at the beginning of the training. Still, after the training finished, the training accuracy of the cnn_3 
architecture was 67, and the validation accuracy of the model was 57, which indicates that the model 
learned the pattern of the data and increased the accuracy of the model.

Moreover, the cnn_4 architecture showed a similar training loss curve and validation accuracy 
vs. training accuracy curve with a slight difference. In this architecture, after step 4, the validation 
accuracy of the model has not improved much, which shows that model started to overfit the data. 
The generalization of the architecture is about to be decreased since the training accuracy of the 
architecture tends to increase in time, but the validation accuracy did not. Lastly, the cnn_5 
architecture showed one of the best generalization performances among the other architectures. The 
main reason for this conclusion is that the validation accuracy vs. training accuracy plot has a similar 
shape. In other words, as the training accuracy increases, the validation accuracy increases too, which 
makes the model prone to different data types since it has a better generalization performance.

**Q4.** The quantity of learnable weights that a machine learning model must modify during training 
is the number of parameters in the model. Generally speaking, a model with more parameters can 
represent more complicated functions, resulting in greater performance on the training set of data. 
This may or may not lead to improved generalization performance on unknown data.

In reality, overfitting, where the model is too closely adapted to the training data and performs 
poorly on new, unforeseen data, can result from having too many parameters. This is due to the 
possibility that the model is catching noise or peculiarities in the training data that do not transfer to 
new data. This is especially problematic when there are many more parameters than training data.
When we examine the architectures, the cnn_5 architecture was one of the best-performing 
models. One of the main reasons for this result is that there are several more parameters in this 
architecture, and in this way, it can learn the pattern better. 

On the other hand, if more than the number of parameters in an architecture is needed, the architecture can show wrong classification and 
generalization performances. So, the optimum number of parameters should be selected to train the 
model with good classification and generalization performances.

**Q5.** The number of layers in a machine learning model architecture is called the model's depth. 
In general, but only sometimes, a deeper architecture can improve performance on both the training 
and test data.

A deeper architecture has the benefit of being able to capture more abstract and complicated 
properties, which can be helpful when learning hierarchical representations of the data. As a result, 
the model can develop its ability to distinguish between classes, resulting in improved performance on 
the training data.

Deeper designs, however, may also be harder to train and may experience disappearing or 
bursting gradients. These problems may make it difficult for the model to pick up helpful information 
and result in subpar performance. Deeper architectures may also be more prone to overfitting, mainly 
if the model is overly complicated compared to the amount of training data.

Techniques like residual or skip connections can be used to reduce vanishing or exploding gradients,
and regularization techniques can be used to avoid overfitting to address these problems.

**Q6.** When I examine the visualizations of the weights, the mlp1 and mlp2 architectures have 
shown a different weight visualization, and the CNN architectures have shown another weight 
visualization. The main difference between these two architecture types is the Max-Pooling layer since 
the size of the matrix decreases because of the Max-Pooling layer, and the output is not interpretable. 
On the other hand, the weight visualizations of the mlp1 and mlp2 architectures have shown patterns.

If I examine these visualizations, the architectures tend to learn the curved figures and some straight 
lines. Especially in the visualizations of the mlp2 architecture, the curves are more precise, and the 
learning of the pattern can be seen easily. Regarding the CNN architectures, it is hard to comment on 
the weight visualizations since the outputs are just 3x3 matrices with white, gray, and black boxes. I 
tried to comment on these visualizations, but the patterns that the CNN architectures learned are not 
easily interpretable.

**Q7.** The mlp1 and mlp2 architectures are specialized to detect the classes that have more curved 
features, namely automobiles, cats, and dogs. It can be concluded from their weight visualizations that 
they have extracted more curved patterns, and the edges of the objects can be seen clearly. When it 
comes to CNN architectures, these models are more generalized architectures, and they are better at 
finding the patterns of the objects in the given input dataset. The hard part is to interpret the weight 
visualizations.

**Q8.** The weights of the mlp1 and mlp2 architectures are more interpretable since their curved 
and straight-line patterns are easily seen from their weight visualizations. On the other hand, CNN 
architectures are hard to comment on.

**Q9.** The multilayer perceptron models, mlp1 and mlp2, have completely connected layers. 
However, mlp2 has a deeper design than mlp1 because it has a second hidden layer. Both models have 
a linear output layer without a bias term and ReLU activation functions between layers.
In terms of completely connected layers and ReLU activation functions, mlp2 is similar to mlp1, 
but it has a more intricate architecture and an additional hidden layer. One hidden layer makes for the 
simpler architecture known as mlp1.

Three convolutional layers are followed by three fully connected layers in the convolutional 
neural network known as the cnn_3. In order to extract features from the input image, each 
convolutional layer applies a set of learnable filters. These features are then passed via non-linear 
activation functions, such as ReLU (Rectified Linear Unit), in order to incorporate non-linearity into the 
model. Following flattening, the output of the last convolutional layer is passed into the fully connected 
layers, which carry out the classification process.

Similar to the cnn_3, the cnn_4 has a fourth convolutional layer, making it a four-layer CNN. 
The additional convolutional layer enables the network to learn more intricate information from the 
input image, potentially enhancing its accuracy in classifying images.

In terms of their structures, cnn3, cnn4, and cnn5 are all similar in that they use convolutional 
layers to extract features from the input image, followed by fully connected layers to perform 
classification. However, cnn4 and cnn5 are more complex than cnn3, with additional convolutional 
layers that allow them to learn more complex features. Additionally, cnn5 has an additional maxpooling layer that helps it to learn translation-invariant features.

Similar to the cnn_4, the cnn_5 is a five-layer CNN, but it adds a convolutional layer before a 
max-pooling layer. The max-pooling layer shrinks the input image's spatial size and aids in the network's 
learning of translation-invariant features, which can increase the network's resistance to changes in 
the input image.

In terms of their structural similarities, CNNs cnn_3, cnn_4, and cnn_5 all use convolutional 
layers to extract features from the input image before performing classification using fully connected 
layers. In contrast, CNNs cnn_4 and cnn_5 are more sophisticated than cnn_3 and can learn more 
intricate characteristics since they have more convolutional layers. A further max-pooling layer on 
cnn_5 aids in the learning of translation-invariant features.

The performance of CNN architectures is superior to MLP architectures. One of the main 
reasons for this is that extracting features and recognizing patterns are more successful than MLP 
architectures thanks to the convolutional layers and max-pool layers in the architecture. In addition to 
these, CNN architectures also have performance differences among themselves. Although the cnn_3architecture is a model with high generalization performance, its training and validation accuracy values are not as high as other CNN architectures. This shows that cnn_3 can have a better training 
score with more training. As for the cnn_4 architecture, although it has higher training accuracy than 
cnn_3, it did not perform well enough in terms of validation accuracy. Apart from these, the cnn_5 
architecture showed a superior result than all other architectures in terms of both training accuracy 
and validation accuracy. This shows that CNN architectures are better at classification than MLP 
architectures. Among the CNN architectures, the most successful model is cnn_5.

**Q10.** I would choose the cnn_5 architecture for the classification task. There are many reasons 
for this, but first of all, I conclude that the cnn_5 architecture, which has a high training accuracy value, 
learns the data well. In addition, seeing that the validation accuracy value does not fall behind the 
training accuracy value shows that the generalization performance of the model is quite good, and it 
will give good results in different data types. In addition to these, the training loss value of the model 
is less than all other models, and thanks to this model, which has a very high test accuracy value, I think 
that I can achieve a good performance even with data that was not included in the training before.



