# ISL-translator

## Requirements 

absl-py==0.7.1
astor==0.7.1
attrs==19.1.0
backcall==0.1.0
bcolz==1.2.1
bleach==1.5.0
certifi==2016.2.28
chardet==3.0.4
Click==7.0
colorama==0.4.1
cycler==0.10.0
decorator==4.4.0
defusedxml==0.5.0
descartes==1.1.0
Django==2.2
django-crispy-forms==1.7.2
entrypoints==0.3
feather-format==0.4.0
Flask==1.0.3
future==0.17.1
fuzzywuzzy==0.17.0
gast==0.2.2
grpcio==1.19.0
h5py==2.9.0
html5lib==0.9999999
idna==2.8
imageio==2.5.0
ipykernel==5.1.0
ipython==7.4.0
ipython-genutils==0.2.0
ipywidgets==7.4.2
itsdangerous==1.1.0
jedi==0.13.3
Jinja2==2.10
joblib==0.13.2
jsonschema==3.0.1
jupyter==1.0.0
jupyter-client==5.2.4
jupyter-console==6.0.0
jupyter-core==4.4.0
Keras==2.2.4
Keras-Applications==1.0.7
keras-contrib==2.0.8
Keras-Preprocessing==1.0.9
kiwisolver==1.0.1
Markdown==3.1
MarkupSafe==1.1.1
matplotlib==3.0.3
mistune==0.8.4
mizani==0.5.4
mock==2.0.0
nbconvert==5.4.1
nbformat==4.4.0
networkx==2.3
notebook==5.7.8
np-utils==0.5.10.0
numpy==1.14.5
opencv-contrib-python==4.0.0.21
palettable==3.2.0
pandas==0.24.2
pandocfilters==1.4.2
parso==0.3.4
patsy==0.5.1
pbr==5.1.3
pickleshare==0.7.5
Pillow==6.0.0
plotnine==0.5.1
progressbar==2.5
prometheus-client==0.6.0
prompt-toolkit==2.0.9
protobuf==3.7.1
ptyprocess==0.6.0
pyarrow==0.14.0
pydub==0.23.1
Pygments==2.3.1
pyparsing==2.3.1
pypiwin32==223
pyrsistent==0.14.11
python-dateutil==2.8.0
pyttsx3==2.71
pytz==2018.9
PyWavelets==1.0.3
pywin32==224
pywinpty==0.5.5
PyYAML==5.1
pyzmq==18.0.1
qtconsole==4.4.3
requests==2.22.0
scikit-image==0.15.0
scikit-learn==0.20.3
scikit-surprise==1.0.6
scipy==1.2.1
seaborn==0.9.0
Send2Trash==1.5.0
six==1.12.0
sklearn-pandas==1.8.0
SpeechRecognition==3.8.1
sqlparse==0.3.0
statsmodels==0.10.1
surprise==0.1
tensorboard==1.8.0
tensorflow-gpu==1.8.0
termcolor==1.1.0
terminado==0.8.2
testpath==0.4.2
tornado==6.0.2
tqdm==4.31.1
traitlets==4.3.2
urllib3==1.25.3
wcwidth==0.1.7
Werkzeug==0.15.2
widgetsnbextension==3.4.2
win-unicode-console==0.5
wincertstore==0.2


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

![alt text](https://github.com/yatharth77/ISL-translator/blob/master/isl.png)

## Dataset
(for data set mail at y.s.saxena7@gmail.com)
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

![alt text](https://github.com/yatharth77/ISL-translator/blob/master/dataset.PNG)

## Methodology

In this section, we would discuss the architectures of various self-developed and pre-trained deep neural networks,
machine learning algorithms and their corresponding performances for the task of hand gesture to audio and audio to
hand gesture recognition. The complete implementation was
done on Keras using Tensorflow as the backend. A pictorial
overview of our entire framework is presented in Fig. 1. The
three individual models are briefly discussed as follows.

![alt text](https://github.com/yatharth77/ISL-translator/blob/master/flow.PNG)

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
![alt text](https://github.com/yatharth77/ISL-translator/blob/master/coil.PNG)
![alt text](https://github.com/yatharth77/ISL-translator/blob/master/coil_out1.PNG)

## Algorithm for formation of Valid English Words from given sequence of alphabets as input

The natural language based output network was developed
for rectifying errors made by the CNN model. The main
motive of this model is to correct the falsely predicted outcomes during ISL-conversation. Thus, a misspelled word
can be corrected by using an algorithm that takes into account the possible words in the English language that can be
formed by the predicted alphabets via intelligently changing a letter or two. Such algorithms are useful in practical terms to overcome the flaws of CNN. A 13-layer CNN
was developed which received these images, with their pix-
5
els scaled between -1 and +1. The neural net was a simple
network comprising of 3×3 convolutional filters followed by
max-pooling. The latter layers consisted dropout (0.3-0.4)
and batch normalisation for avoiding any overfitting. Adam
optimiser with a learning rate of 0.0002 was used to minimize the categorical cross-entropy loss function. The softmax layer provided output as 26 probabilities, each corresponding to the output being that particular alphabet. Exploiting this characteristic of the softmax layer, we calculated
the total probability for a given word, which is a collection
of alphabets as the sum of all the probabilities of the highest
predicted output for that alphabet.
For example, if the word that a user inputs alphabet of
‘cat’, then for each letter ‘c’, ‘a’ and ‘t’, the probabilities for
the top-3 predicted letters will be saved. This will be the
overall probability of the word being ‘cat’. Now, if the output probabilities that the CNN provided with respect to each
letter corresponded to ‘cet’, then this word will be searched
through a corpora of length=3 in the English dictionary. If no
such word exists, it will change the letters (one at a time) by
the next highest probable letter, and check it again in the dictionary. If such a word exists, then it is stored along with it’s
total probability. The model output the word with the highest
probability belonging in the English dictionary as the final
prediction. This model works on the idea that if a user wants
to converse in finger-spelled ISL, he/she is likely to depict
a word that exists in the English dictionary (apart from unusual proper nouns).

![alt text](https://github.com/yatharth77/ISL-translator/blob/master/tree.png)

# Interface 

## WebCam Input
![alt text](https://github.com/yatharth77/ISL-translator/blob/master/O_BTP.jpeg)
## Gesture Based KeyBoard
![alt text](https://github.com/yatharth77/ISL-translator/blob/master/Capture1.PNG)
## Emergency Messaging Facility Page
![alt text](https://github.com/yatharth77/ISL-translator/blob/master/Capture3.PNG)


