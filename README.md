# ISL-translator

## Abstract

Sign language is the language of the deaf and mute. However, this particular population of the
world is unfortunately overlooked as sign language is not understood by the majority hearing population. In
this paper, an extensive comparative analysis of various gesture recognition techniques involving convolutional
neural networks and machine learning algorithms have been discussed and tested for real-time accuracy. Three
models: a pre-trained VGG16 with fine-tuning, VGG16 with transfer learning and a hierarchical neural network
were analysed based on number of trainable parameters. These models were trained on a self-developed dataset
consisting images of Indian Sign Language (ISL) representation of all 26 English alphabets. The performance
evaluation was also based on practical application of these models, which was simulated by varying lighting and
background environments. Out of the three, the Hierarchical model outperformed the other two models to give the
best accuracy of 98.52% for one-hand and 97% for two-hand gestures. Thereafter, a conversation interface was
built in Django using the best model (viz. hierarchical neural networks) for real-time gesture to speech conversion
and vice versa. This publicly accessible interface can be used by anyone who wishes to learn or converse in ISL.

## Dataset
The dataset used for this work was based on ISL. According
to the best of the knowledge of the authors, there does not
exist an authentic and complete dataset for all the 26 alphabets of English language for ISL. Our dataset was manually
prepared by clicking various images of each finger-spelled
alphabet and applying different forms of data augmentation
techniques. At the end, the dataset contained over 1,50,000
images of all 26 categories. There were approximately 5,500
images of each alphabet. To keep the data consistent, the
same background was used for most of the images. Also, the
images were clicked in different lighting conditions to train a
robust model resistant of any such changes in the surroundings. The images in this dataset were clicked by a Redmi
Note 5 Pro, 20 megapixel camera. All the RGB images were
resized to 144×144 pixels per image so as to remove the possibility of varying sizes. Fig.2 shows a few sample images from this dataset.

## Methodology

In this section, we would discuss the architectures of various self-developed and pre-trained deep neural networks,
machine learning algorithms and their corresponding performances for the task of hand gesture to audio and audio to
hand gesture recognition. The complete implementation was
done on Keras using Tensorflow as the backend. A pictorial
overview of our entire framework is presented in Fig. 1. The
three individual models are briefly discussed as follows.

• **Pre-trained VGG16 Model**: Under this approach, the
gestures were classified using a pre-trained VGG16
model based on the Imagenet dataset. We truncated
its last layer and then added custom designed layers to
provide a baseline comparison with the state of the art
networks.

• **Natural Language Based Output Networks**: For this
model, a Deep Convolutional Neural Network (DCNN)
with 26 categories was developed. Later, the output
was fed to an English Corpora based model for eradicating any errors during classification. This process
was based on the probability of the occurrence of the
particular word in the English vocabulary. Moreover,
only the top-3 accuracy scores provided by the neural
network was considered in this model.

• **Hierarchical Network**: Our final approach comprises
of a novel hierarchical model for classification which
resembles a tree-like structure. It involves initially classifying gestures into two categories (one-hand or twohand), and subsequently feeding them into further deep
neural networks. The corresponding outputs were utilized for categorizing them into the 26 English alphabets.


## Experimental Results

All the three networks were tested on our dataset of around
1,49,568 images. The testing dataset consists of hand images
clicked on a black background. The images were augmented,
resized and pre-processed as per the network requirements
mentioned earlier. The pre-trained VGG16 model with transfer learning produced an accuracy of 53% and the fine-tuned
model resulted in an accuracy of 94.52%. The reason for low
accuracy in VGG16 can be attributed to the fact that it was
initially trained for 1000 categories. However, the present
model works on 26 hand gestures and out of them, quite a
few are very similar to one another. The hierarchical model
resulted in a training loss of 0.0016, thus resulting in a training accuracy of 99% and a validation accuracy of 98.52% for
categorising one-hand features and 99% training accuracy
and 97% validation accuracy for two-hand features respectively. The SVM used in hierarchical model for categorising into one-hand and two-hand gestures, combined with the
HOG features, produced and accuracy of 96.79%. This result
is significantly better than any other machine learning algorithm for the 26 classes. The confusion matrix in Fig. 6 segregates the result obtained on the 6085 test samples into truepositives, true-negatives, false-positives and false-negatives.

![alt text](https://github.com/yatharth77/ISL-translator/blob/master/O_BTP.jpeg)
