#Facial expression recognition using SVM

Extract face landmarks using Dlib and train a multi-class SVM classifier to recognize facial expressions (emotions).


##Motivation
Fer2013 images are not aligned and it's difficult to classify facial expression from it.

The best accuracy for Fer2013 (as I know) is 67%, the author trained a Convolutional Neural Network during several hours in a powerful GPU to obtain this results.
Let's try a much simpler (and faster) approach by extracting Face Landmarks and HOG features and feed them to a multi-class SVM classifier.


##Results:

/--------------------------------------------------------\
|       Features        |  7 emotions   |   5 emotions   |
|--------------------------------------------------------|
| HoG features          |     29.0%     |      34.4%     |
| Face landmarks        |     39.2%     |      46.9%     |
| Face landmarks + HOG  |     48.2%     |      55.0%     |
|--------------------------------------------------------|
| Max training time     |    443 sec    |     288 sec    |
\--------------------------------------------------------/

While the training time is very short compared to CNN, we lost 19% in accuracy compared to the actual best result that uses CNN.

Note: It's possible to obtain better results by changing parameters. One may implement a hyperparameters search to find the best parameters.

##How to use

1. Extract "fer2013_landmarks+hog.zip" file

2. Install dependencies

```
pip install Numpy
pip install argparse
pip install sklearn
```

3. Train model:

```
python train.py --train=yes
```

4. Evaluate model

If you have already a pretrained model

```
python train.py --evaluate=yes
```

5. Train and evaluate [instead of step 3 and 4]

```
python train.py --train=yes --evaluate=yes 
```
