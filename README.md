# ISL-translator

Abstract

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

#Methodology

In this section, we would discuss the architectures of various self-developed and pre-trained deep neural networks,
machine learning algorithms and their corresponding performances for the task of hand gesture to audio and audio to
hand gesture recognition. The complete implementation was
done on Keras using Tensorflow as the backend. A pictorial
overview of our entire framework is presented in Fig. 1. The
three individual models are briefly discussed as follows.

• Pre-trained VGG16 Model: Under this approach, the
gestures were classified using a pre-trained VGG16
model based on the Imagenet dataset. We truncated
its last layer and then added custom designed layers to
provide a baseline comparison with the state of the art
networks.

• Natural Language Based Output Networks: For this
model, a Deep Convolutional Neural Network (DCNN)
with 26 categories was developed. Later, the output
was fed to an English Corpora based model for eradicating any errors during classification. This process
was based on the probability of the occurrence of the
particular word in the English vocabulary. Moreover,
only the top-3 accuracy scores provided by the neural
network was considered in this model.

• Hierarchical Network: Our final approach comprises
of a novel hierarchical model for classification which
resembles a tree-like structure. It involves initially classifying gestures into two categories (one-hand or twohand), and subsequently feeding them into further deep
neural networks. The corresponding outputs were utilized for categorizing them into the 26 English alphabets.
