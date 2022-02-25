# ... Step. 0 Libraries
from re import M
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random 
import glob

import wandb
from wandb.keras import WandbCallback

wandb.login()

# ... TODO: Record these variables
name = "name"
project_name = "project"

# ... TODO: Put path to folder for epoch_cubes .npy files
folder_cubes = "/path"

# ... TODO: Put path to folder for epoch_cubes labels
folder_cube_labels = "/path"

# ... TODO: Give Wandb.init your config
run = wandb.init(project=project_name, entity=name, config={
    "learning_rate": 0.0001,
    "epochs": 100,
    "batch_size": 30,
    "loss_function": "binary_crossentropy",
    "optimizer": "Adam",
    "filters1": 18,
    "filters2": 36,
    "kernel_size": (3,3,3),
    "dropout": 0.5,
    "units": 128
    })

config = wandb.config

# ... Get labels for the cubes
def get_labels(cube_path):

    # Takes participant number
    key = cube_path.split("-")[2]

    # Loading in a folder with two .npy file per person (cubes, labels)
    labels = np.load(f"{folder_cube_labels}/3D_epoch_cube_labels_NP-Ph2-{key}")

    return labels

# ... Make list of cube files for all participants:
cube_list = glob.glob(f"{folder_cubes}/3D_epoch_cubes_*.npy")

# ... Make test set (taking it out for later test)
test_paths = [
    f"{folder_cubes}/3D_epoch_cubes_NP-Ph2-306_RESHAPED_RefSpe_visual.npy",
    f"{folder_cubes}/3D_epoch_cubes_NP-Ph2-104_RESHAPED_RefSpe_visual.npy",
    f"{folder_cubes}/3D_epoch_cubes_NP-Ph2-123_RESHAPED_RefSpe_visual.npy",
    f"{folder_cubes}/3D_epoch_cubes_NP-Ph2-125_RESHAPED_RefSpe_visual.npy"]

test_cubes = np.load(f"{test_paths[0]}")
test_labels = get_labels(test_paths[0])
cube_list_no_test = cube_list.copy()
cube_list_no_test.remove(test_paths[0])

# ... Looping through - concatenating everything from 1: behind cube number 0
for i in test_paths[1:]:
    test_cube = np.load(f"{i}")
    test_label = get_labels(i)
    test_cubes = np.concatenate((test_cubes,test_cube),axis = 0)
    test_labels = np.concatenate((test_labels,test_label),axis = 0)
    cube_list_no_test.remove(i)

# ... Create function: 
# Creating validation set from 3 participant's cubes (3 people in the val_dataset)
def data_split(cube_list_no_test):

    new_cube_list = cube_list_no_test.copy()

    validation_paths = random.sample(new_cube_list, 3)

    validation_cubes = np.load(f"{validation_paths[0]}")

    validation_labels = get_labels(validation_paths[0])

    print(validation_paths)

    new_cube_list.remove(validation_paths[0])

    # Looping through validation paths, generating one big cube for the validation dataset
    for i in validation_paths[1:]:
        validation_cube = np.load(f"{i}")
        validation_label = get_labels(i)
        validation_cubes = np.concatenate((validation_cubes,validation_cube),axis = 0)
        validation_labels = np.concatenate((validation_labels,validation_label),axis = 0)
        new_cube_list.remove(i) # Removing the val-participants from the training list


    # Create the training set by concatenating the rest of the epoch arrays:
    training_cube1 = new_cube_list[0]
    print(new_cube_list)
    training_cubes = np.load(f"{training_cube1}")
    training_labels = get_labels(training_cube1)
    for cube in new_cube_list[1:]:
        cube_3D = np.load(f"{cube}") 
        cube_label = get_labels(cube)
        training_cubes = np.concatenate((training_cubes,cube_3D),axis = 0)
        #print(training_cubes.shape)
        training_labels = np.concatenate((training_labels,cube_label),axis = 0)
        #print(training_labels.shape)
    return training_cubes, training_labels, validation_cubes, validation_labels

# ... Using the function - splitting the set in validation, train (test-set is made above)
samples_no_test = data_split(cube_list_no_test)

# ... Definitions of functions for changing labels and normalizing
# Change labels
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

# ... Renaming the label lists (verbatim - what they are)
training_labels = change_labels_verbatim(samples_no_test[1])
validation_labels = change_labels_verbatim(samples_no_test[3])
test_labels = change_labels_verbatim(test_labels)

# ... Normalize the data
training_data = norm_func(samples_no_test[0])
validation_data = norm_func(samples_no_test[2])
testing_data = norm_func(test_cubes)

# ... Renaming the label lists (binary)
new_labels_train = []
for label in training_labels[0:]:
    if label == 'Condition1':
        newlabel = 0
    else:
        newlabel = 1
    new_labels_train.append(newlabel)

new_labels_validation = []
for label in validation_labels[0:]:
    if label == 'Condition1':
        newlabel = 0
    else:
        newlabel = 1
    new_labels_validation.append(newlabel)

new_labels_test = []
for label in test_labels[0:]:
    if label == 'Condition1':
        newlabel = 0
    else:
        newlabel = 1
    new_labels_test.append(newlabel)

# ... Using tensorslices to create dataset with labels 
train_dataset = tf.data.Dataset.from_tensor_slices((training_data, new_labels_train)) 
validation_dataset = tf.data.Dataset.from_tensor_slices((validation_data, new_labels_validation))
test_dataset = tf.data.Dataset.from_tensor_slices((testing_data, new_labels_test))

# ... Batching and shuffling
BATCH_SIZE = config.batch_size
SHUFFLE_BUFFER_SIZE = len(train_dataset) # shuffle randomly shuffles the elements. 
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_dataset = validation_dataset.shuffle(len(validation_dataset)).batch(BATCH_SIZE) 
test_dataset = test_dataset.batch(BATCH_SIZE)

# ... Define a model 
def get_model(width=36, height=36, depth=81):
    inputs = keras.Input((height, width, depth, 3))

    x = layers.Conv3D(filters=config.filters1, kernel_size=config.kernel_size, activation="relu", kernel_regularizer='l2')(inputs) 
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dropout)(x) 

    x = layers.Conv3D(filters=config.filters2, kernel_size=config.kernel_size, activation="relu", kernel_regularizer='l2')(x) 
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.dropout)(x) 

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(config.units, activation="relu")(x) 

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

tf.keras.backend.clear_session()

# ... Build model.
model = get_model(width=36, height=36, depth=81)
model.summary()

# ... Compile model
model.compile(
    loss=config.loss_function,
    optimizer=keras.optimizers.Adam(config.learning_rate),
    metrics=["acc"],
)

# ... Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "best_model_3d_image_classification.h5", save_best_only=True, monitor = 'val_accuracy', mode = 'max', verbose = 1 
)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", mode = 'min', patience=30, verbose = 1)  

# We train with our beloved model.fit
# Notice WandbCallback is used as a regular callback
# We again use config
_ = model.fit(train_dataset,
          epochs=config.epochs, 
          batch_size=config.batch_size,
          validation_data=(validation_dataset),
          callbacks=[WandbCallback(),early_stopping_cb])

run.finish()