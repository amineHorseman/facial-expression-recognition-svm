#Facial expression recognition using SVM

Extract face landmarks using Dlib and train a multi-class SVM classifier to recognize facial expressions (emotions).

##Motivation
Fer2013 images are not aligned and it's difficult to classify facial expression on it.
The best accuracy for Fer2013 (as I know) is 67%, the author trained a Convolutional Neural Network during several hours in a powerful GPU to obtain this results.
Let's try a much simpler (and faster) approach by extracting first face landmarks and train them on a multi-class SVM:

##Dependencies
- Numpy
- Argparse
- Sklearn

##How to use

1. Extract "fer2013_landmarks.zip" file

2. Convert Fer2013 and extract landmarks

'''
python convert_fer2013_to_npy.py
'''

3. Train model:

'''
python train.py --train=yes
'''

4. Evaluate model

If you have already a pretrained model

'''
python train.py --evaluate=yes
'''

5. Train and evaluate [instead of step 3 and 4]

'''
python train.py --train=yes --evaluate=yes 
'''

##Results:

- Training time: 80 seconds
- Accuracy using 5 emotions = 46.9%
- Accuracy using 7 emotions = 39.2%

While the training time is very nice, we lost 28% in accuracy compared to the actual best result using CNN.

I wonder which accuraccy we'll get if we feed the extracted landmarks to a CNN?

