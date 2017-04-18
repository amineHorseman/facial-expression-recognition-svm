import numpy as np
import pandas as pd
import os
import scipy.misc
import dlib
import cv2

from skimage.feature import hog

image_height = 48
image_width = 48
ONE_HOT_ENCODING = True
SAVE_IMAGES = False
GET_LANDMARKS = True
GET_HOG_FEATURES = False
FILTERED_LABELS = [0, 1, 2, 5]  # 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
IMAGES_PER_LABEL = 1000000000

print "preparing"
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
original_labels = [0, 1, 2, 3, 4, 5, 6]
new_labels = list(set(original_labels) ^ set(FILTERED_LABELS))
nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))

def get_landmarks(image, rects):
    # this function have been copied from http://bit.ly/2cj7Fpq
    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces
    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def get_new_label(label, one_hot_encoding=False):
    if one_hot_encoding:
        new_label = new_labels.index(label)
        label = list(np.zeros(len(new_labels), 'uint8'))
        label[new_label] = 1
        return label
    else:
        return new_labels.index(label)

print "importing csv file"
data = pd.read_csv('fer2013.csv')

for category in data['Usage'].unique():
    print "converting set: " + category + "..."
    # create folder
    if not os.path.exists(category):
        os.makedirs(category)
    
    # get samples and labels of the actual category
    category_data = data[data['Usage'] == category]
    samples = category_data['pixels'].values
    labels = category_data['emotion'].values
    
    # get images and extract features
    images = []
    labels_list = []
    landmarks = []
    hog_features = []
    hog_images = []
    for i in xrange(len(samples)):
        try:
            if labels[i] not in FILTERED_LABELS and nb_images_per_label[get_new_label(labels[i])] < IMAGES_PER_LABEL:
                image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
                images.append(image)
                if SAVE_IMAGES:
                    scipy.misc.imsave(category + '/' + str(i) + '.jpg', image)
                if GET_HOG_FEATURES:
                    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                                            cells_per_block=(1, 1), visualise=True)
                    hog_features.append(features)
                    hog_images.append(hog_image)
                if GET_LANDMARKS:
                    scipy.misc.imsave('temp.jpg', image)
                    image2 = cv2.imread('temp.jpg')
                    face_rects = [dlib.rectangle(left=1, top=1, right=47, bottom=47)]
                    face_landmarks = get_landmarks(image2, face_rects)
                    landmarks.append(face_landmarks)            
                labels_list.append(get_new_label(labels[i], one_hot_encoding=ONE_HOT_ENCODING))
                nb_images_per_label[get_new_label(labels[i])] += 1
        except Exception as e:
            print "error in image: " + str(i) + " - " + str(e)

    np.save(category + '/images.npy', images)
    if ONE_HOT_ENCODING:
        np.save(category + '/labels.npy', labels_list)
    else:
        np.save(category + '/labels_categorical.npy', labels_list)
    if GET_LANDMARKS:
        np.save(category + '/landmarks.npy', landmarks)
    if GET_HOG_FEATURES:
        np.save(category + '/hog_features.npy', hog_features)
        np.save(category + '/hog_images.npy', hog_images)