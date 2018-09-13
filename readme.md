# Facial expression recognition using SVM

Extract face landmarks using Dlib and train a multi-class SVM classifier to recognize facial expressions (emotions).


## Motivation:

The task is to categorize people images based on the emotion shown by the facial expression. 
To train our model, we want to use Fer2013 datset that contains 30,000 images of expressions grouped in seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise and Neutral.
The problem is that Fer2013 images are not aligned and it's difficult to classify facial expressions from it.
The state-of-art accuracy achieved in this dataset, as far as I know, is 75.2% (refer to: *Christopher Pramerdorfer, Martin Kampel. "Facial Expression Recognition using Convolutional Neural Networks: State of the Art". arXiv:1612.02903v1, 2016*), a Convolutional Neural Network was used during several hours on GPU to obtain these results.
Lets try a much simpler (and faster) approach by extracting Face Landmarks + HOG features and feed them to a multi-class SVM classifier. The goal is to get a quick baseline for educational purpose, if you want to achieve better results please refer to Pramerdorfer's paper. 


## Classification Results :

|       Features                          |  7 emotions   |   5 emotions   |
|-----------------------------------------|---------------|----------------|
| HoG features                            |     29.0%     |      34.4%     |
| Face landmarks                          |     39.2%     |      46.9%     |
| Face landmarks + HOG                    |     48.2%     |      55.0%     |
| Face landmarks + HOG on slinding window |     50.5%     |      59.4%     |

As expected, the Deep Learning approaches achieve better results (compare results with [Facial Expressions Recognition using CNN](https://github.com/amineHorseman/facial-expression-recognition-using-cnn))

The SVM training time was about ~400 seconds on an i7 2.8Ghz CPU, for the last experiment (sliding window) the training time reached 2060 seconds.

For the experiments with 5 emotions, the following expressions was used: Angry, Happy, Sad, Surprise, Neutral.

The code was tested for python 2.7 and 3.6.

## How to use

1. Download Fer2013 dataset and the Face Landmarks model

    - [Kaggle Fer2013 challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
    - [Dlib Shape Predictor model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

2. Unzip the downloaded files

    And put the files `fer2013.csv` and `shape_predictor_68_face_landmarks.dat` in the root folder of this package.

3. Install dependencies

    ```
    pip install numpy
    pip install argparse
    pip install sklearn
    pip install scikit-image
    pip install pandas
    pip install hyperopt
    pip install dlib
    ```

4. Convert the dataset to extract Face Landmarks and HOG Features

    ```
    python convert_fer2013_to_images_and_landmarks.py
    ```

    You can also use these optional arguments according to your needs:
    `-j`, `--jpg` (yes|no): **save images as .jpg files (default=no)**
    `-l`, `--landmarks` *(yes|no)*: **extract Dlib Face landmarks (default=yes)**
    `-ho`, `--hog` (yes|no): **extract HOG features (default=yes)**
    `-hw`, `--hog_windows` (yes|no): **extract HOG features using a sliding window (default=yes)**
    `-o`, `--onehot` (yes|no): **one hot encoding (default=no)**
    `-e`, `--expressions` (list of numbers): **choose the faciale expression you want to use: *0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral* (default=0,1,2,3,4,5,6)**

    Example
    ```
    python convert_fer2013_to_images_and_landmarks.py --landmarks=yes --hog=no --how_windows=no --jpg=no --onehot=no --expressions=1,3,4
    ```

5. Train the model

    ```
    python train.py --train=yes
    ```

6. Evaluate the model

    If you have already a pretrained model

    ```
    python train.py --evaluate=yes
    ```

7. Train and evaluate [instead of step 5 and 6]

    ```
    python train.py --train=yes --evaluate=yes 
    ```

8. Customize the training parameters:

    Feel free to change the values of the parameters in the `parameters.py` file according to your needs.

9. Find the best hyperparameters (using hyperopt):

    ```
    python optimize_parameters.py --max_evals=15
    ```
    The argument `max_evals` specifies the number of combinaisions hyperopt will try.
    After finding the best hyperparameters, you should change the corresponding values in `parameters.py` and retrain the model.
    N.B: the accuracies displayed during hyperoptimization are for validation_set only (not test_set)
    
## Contributing

Some ideas for interessted contributors:
- Add other features extraction techniques?
- Predict expression from a .jpg|.png file?
