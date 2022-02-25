# ... Step. 0 Libraries
from re import M
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random 
import glob
from numpy import loadtxt
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import pandas as pd


# -- TODO: Put path to folder for epoch_cubes .npy files
folder_cubes = "/path"

# -- TODO: Put path to folder for epoch_cubes labels
folder_cube_labels = "/path"

# -- Get labels for the cubes
def get_labels(cube_path):

    # Takes participant number
    key = cube_path.split("-")[2]

    # Loading in a folder with two .npy file per person (cubes, labels)
    labels = np.load(f"{folder_cube_labels}/3D_epoch_cube_labels_NP-Ph2-{key}")

    return labels

# -- Make list of cube files for all participants:
cube_list = glob.glob(f"{folder_cubes}/EpochCubes_SpeakThink/3D_epoch_cubes_*.npy")

# -- Make test set
test_paths = [
    f"{folder_cubes}/3D_epoch_cubes_NP-Ph2-306_RESHAPED_THINK_SPEAK_visual.npy",
    f"{folder_cubes}/3D_epoch_cubes_NP-Ph2-104_RESHAPED_THINK_SPEAK_visual.npy",
    f"{folder_cubes}/3D_epoch_cubes_NP-Ph2-123_RESHAPED_THINK_SPEAK_visual.npy",
    f"{folder_cubes}/3D_epoch_cubes_NP-Ph2-125_RESHAPED_THINK_SPEAK_visual.npy"]

test_cubes = np.load(f"{test_paths[0]}")
test_labels = get_labels(test_paths[0])
cube_list_no_test = cube_list.copy()
cube_list_no_test.remove(test_paths[0])

# -- Looping through - concatenating everything from 1: behind cube number 0
for i in test_paths[1:]:
    test_cube = np.load(f"{i}")
    test_label = get_labels(i)
    test_cubes = np.concatenate((test_cubes,test_cube),axis = 0)
    test_labels = np.concatenate((test_labels,test_label),axis = 0)
    cube_list_no_test.remove(i)

# -- Definitions of functions for changing labels and normalizing
def change_labels_verbatim(label_array):
    verb_label_array = []
    for label in label_array: 
        if label in (46,47,48,55,56,57,64,65,66,73,74,75): 
            verb_label_array.append("Condition1")
        elif label in (49,50,51,58,59,60,67,68,69,76,77,78):
            verb_label_array.append("Condition2")
    return verb_label_array

# Normalizing function
def norm_func(array):
    norm_array = array/255
    return norm_array

# -- Renaming the label lists (verbatim - what they are)
test_labels = change_labels_verbatim(test_labels)

# -- Normalizing
testing_data = norm_func(test_cubes)

# -- Relabel
new_labels_test = []
for label in test_labels[0:]:
    if label == 'Condition1':
        newlabel = 0
    else:
        newlabel = 1
    new_labels_test.append(newlabel)

# -- Using tensorslices to create dataset with labels 
test_dataset = tf.data.Dataset.from_tensor_slices((testing_data, new_labels_test))
test_dataset = test_dataset.batch(20)

# -- Testing
model = load_model('/path/to/saved/model/from/sweep.h5') 
loss, accuracy = model.evaluate(test_dataset)
print('Test Error Rate: ', round((1 - accuracy) * 100, 2))
y_pred_bottom = (model.predict(test_dataset).ravel()>0.5)+0
print(f'f1-score is {f1_score(new_labels_test, y_pred_bottom, average="weighted")}')

confusion_matrix_bottom = confusion_matrix(new_labels_test,y_pred_bottom)
print(confusion_matrix_bottom)