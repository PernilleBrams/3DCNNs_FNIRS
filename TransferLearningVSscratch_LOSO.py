
dirpath = "path/to/files"

# Libraries
from re import M
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random 
import glob
from numpy import loadtxt
from keras.models import load_model
import sys 
import os.path
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import pandas as pd

# Participant paths to /.npy files
participant_paths = [
    f"{dirpath}3D_epoch_cubes_NP-Ph2-102_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-104_RESHAPED_RefSpe_visual.npy", 
    f"{dirpath}3D_epoch_cubes_NP-Ph2-105_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-107_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-109_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-110_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-111_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-112_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-118_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-122_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-123_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-124_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-125_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-128_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-129_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-200_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-201_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-203_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-204_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-300_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-302_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-303_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-304_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-306_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-400_RESHAPED_RefSpe_visual.npy",
    f"{dirpath}3D_epoch_cubes_NP-Ph2-402_RESHAPED_RefSpe_visual.npy"]


# -- Functions needed for running the model
## Change labels to verbatim
def change_labels_verbatim(label_array):
    verb_label_array = []
    for label in label_array: 
        if label in (46,47,48,55,56,57,64,65,66,73,74,75): 
            verb_label_array.append("Condition1")
        elif label in (49,50,51,58,59,60,67,68,69,76,77,78):
            verb_label_array.append("Condition2")
    return verb_label_array

# Get labels for participant
def get_labels(cube_path):
    key = cube_path.split("-")[2]
    print(key)
    labels = np.load(f"{dirpath}/3D_epoch_cube_labels_NP-Ph2-{key}")
    return labels

# Model architecture
def get_model(width=36, height=36, depth=81):

    inputs = keras.Input((height, width, depth, 3))

    x = layers.Conv3D(filters=18, kernel_size=6, activation="relu", kernel_regularizer='l2')(inputs) 
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x) 

    x = layers.Conv3D(filters=128, kernel_size=6, activation="relu", kernel_regularizer='l2')(x) 
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x) 

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(256, activation="relu")(x) 

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model

# Transfer learning type
transfer_learning_type = ['feature_extraction','finetuning','finetuning_all','from_scratch'] 


# - Models
# Participant loop for all participants, 20 epochs
for participant in participant_paths:

    # Get participant ID
    part_name = os.path.basename(participant)

    # Define the k-fold
    kfold = KFold(n_splits=5, shuffle=True) # NOTE 
    
    # Find data for the participant to make the K-fold split:
    samples_3D = np.load(participant, allow_pickle=True)  
    samples_normed = samples_3D/255

    labels_3D = get_labels(participant) 
       
    fold_no = 1

    # List of dataframes
    participant_dataframes = []

    # Get basename (just the part. number)
    basename = os.path.basename(participant)
    basename = basename.split("-")[2].partition('_')[0] 

    # Making txt files - it will create file if it does not exists in library    
    file_object_feat = open(f"FE_{basename}.txt", "w+")
    file_object_feat.write(f"###### --- FEATURE EXTRACTION for participant {basename}--- #####"+ '\n'+ '\n')
    file_object_feat.close()
  
    file_object_fine = open(f"FT_{basename}.txt", "w+")
    file_object_fine.write(f"###### --- FINETUNING for participant {basename}--- #####"+ '\n'+ '\n')
    file_object_fine.close()
   
    file_object_fine_a = open(f"FTA_{basename}.txt", "w+")
    file_object_fine_a.write(f"###### --- FINETUNING (ALL LAYERS) for participant {basename}--- #####"+ '\n'+ '\n')
    file_object_fine_a.close()
   
    file_object_scratch = open(f"Scratch_{basename}.txt", "w+")
    file_object_scratch.write(f"###### --- SCRATCH MODEL for participant {basename}--- #####"+ '\n'+ '\n')
    file_object_scratch.close()

    # --- K_FOLD VALIDATION
    for train, test in kfold.split(samples_normed, labels_3D):
        print(f"########    FOLD: Starting on fold {fold_no} ########### ")

        for type in transfer_learning_type:
            print(f"########    TYPE TL: Starting on {type} ########### ")
    # Creating the name for the output file

            # --- GET THE PRETRAINED MODEL ---
            key_model = participant.split("-")[2]

            # Loading in the right model
            source_model = load_model(f"model_3D_epoch_cubes_NP-Ph2-{key_model}.h5")

            print(f"Source model loaded in for participant {participant} is 'model_3D_epoch_cubes_NP-Ph2-{key_model}.h5'")
            
            # Create the new model by freezing the Conv layers and add new dense layers:
            model = keras.Sequential()

            for layer in source_model.layers[:-2]:
            # Go through until last two layers to remove the Dense 1-neuron output layer and the second to last cause we only want the features
                model.add(layer)

            ## model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(256, activation="relu"))
            model.add(keras.layers.Dense(1, activation="sigmoid"))

            model.compile(
                loss='binary_crossentropy',
                optimizer=keras.optimizers.Adam(0.00001),
                metrics=["acc"],
                )

            model.build(input_shape = (None, 36,36,81,3)) 

            # Get a summary of the model
            model.summary()

            if type == 'feature_extraction':
                ## Freeze all layers
                model.get_layer(index = 0).trainable = False
                model.get_layer(index = 1).trainable = False
                model.get_layer(index = 2).trainable = False
                model.get_layer(index = 3).trainable = False
                model.get_layer(index = 4).trainable = False
                model.get_layer(index = 5).trainable = False
                model.get_layer(index = 6).trainable = False
                model.get_layer(index = 7).trainable = False
                model.get_layer(index = 8).trainable = False
                model.get_layer(index = 9).trainable = False

                # Renaming the label lists (verbatim - what they are)
                training_labels = change_labels_verbatim(labels_3D[train])
                test_labels = change_labels_verbatim(labels_3D[test])

                # Renaming the label lists (binary - manual solution)
                new_labels_train = []
                for label in training_labels[0:]:
                    if label == 'Condition1':
                        newlabel = 0
                    else:
                        newlabel = 1
                    new_labels_train.append(newlabel)

                new_labels_test = []
                for label in test_labels[0:]:
                    if label == 'Condition1':
                        newlabel = 0
                    else:
                        newlabel = 1
                    new_labels_test.append(newlabel)

                train_dataset = tf.data.Dataset.from_tensor_slices((samples_normed[train], new_labels_train)) 
                test_dataset = tf.data.Dataset.from_tensor_slices((samples_normed[test], new_labels_test))

                BATCH_SIZE = 4

                SHUFFLE_BUFFER_SIZE = len(samples_normed[train]) # shuffle randomly shuffles the elements. 
                train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
                test_dataset = test_dataset.batch(BATCH_SIZE)

                history = model.fit(train_dataset,
                    epochs=20,
                    batch_size=4)

                model.save(f"model_FE{participant}_Fold{fold_no}.h5")

                # Saving dataframe of training acc/loss
                hist_feature_extraction_df = pd.DataFrame(history.history)

                # Saving test metrics
                loss, accuracy = model.evaluate(test_dataset) # NOTE:
                y_pred = (model.predict(test_dataset).ravel()>0.5)+0
                confusion_matrix_tl = confusion_matrix(new_labels_test,y_pred)
                f1score = f1_score(new_labels_test, y_pred, average="weighted")

                # SCORES FOR LATER IN TXT FILE
                quote_acc = f"Test Accuracy in {fold_no} is {accuracy}"
                quote_testerror = f"Test Error Rate in {fold_no} is {1-accuracy}" 
                quote_loss = f"Test Loss in {fold_no} is {loss} "
                quote_confusionmatrix = f"Confusion matrix in {fold_no} is {confusion_matrix_tl}"
                quote_f1score = f"F1_score in {fold_no} is {f1score}"

                # Open txt and put them in with access mode 'a'
                file_object_feat = open(f"FE_{basename}.txt", 'a')

                # Append the objects
                file_object_feat.write(quote_acc + '\n' + quote_testerror + '\n' + quote_loss+ '\n' +quote_confusionmatrix + '\n'+ quote_f1score+ '\n'+ '\n')
                file_object_feat.close()



            elif type == 'finetuning':
                ## Unfreeze last conv stage layers
                model.get_layer(index = 0).trainable = False
                model.get_layer(index = 1).trainable = False
                model.get_layer(index = 2).trainable = False
                model.get_layer(index = 3).trainable = False

                # Renaming the label lists (verbatim - what they are)
                training_labels = change_labels_verbatim(labels_3D[train])
                test_labels = change_labels_verbatim(labels_3D[test])

                # Renaming the label lists (binary - manual solution)
                new_labels_train = []
                for label in training_labels[0:]:
                    if label == 'Condition1':
                        newlabel = 0
                    else:
                        newlabel = 1
                    new_labels_train.append(newlabel)

                new_labels_test = []
                for label in test_labels[0:]:
                    if label == 'Condition1':
                        newlabel = 0
                    else:
                        newlabel = 1
                    new_labels_test.append(newlabel)

                train_dataset = tf.data.Dataset.from_tensor_slices((samples_normed[train], new_labels_train)) 
                test_dataset = tf.data.Dataset.from_tensor_slices((samples_normed[test], new_labels_test))

                BATCH_SIZE = 4

                SHUFFLE_BUFFER_SIZE = len(samples_normed[train]) 
                train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
                test_dataset = test_dataset.batch(BATCH_SIZE)

                history = model.fit(train_dataset,
                    epochs=20,
                    batch_size=4)

                model.save(f"model_FT{participant}_Fold{fold_no}.h5")

                # Saving dataframe of train acc/loss
                hist_finetuning_df = pd.DataFrame(history.history)

                # Saving test metrics
                loss_ft, accuracy_ft = model.evaluate(test_dataset) # NOTE:
                y_pred = (model.predict(test_dataset).ravel()>0.5)+0
                confusion_matrix_tl_ft = confusion_matrix(new_labels_test,y_pred)
                f1score_ft = f1_score(new_labels_test, y_pred, average="weighted")

                # SCORES FOR LATER IN TXT FILE
                quote_acc = f"Test Accuracy in {fold_no} is {accuracy_ft}"
                quote_testerror = f"Test Error Rate in {fold_no} is {1-accuracy_ft}" 
                quote_loss = f"Test Loss in {fold_no} is {loss_ft} "
                quote_confusionmatrix = f"Confusion matrix in {fold_no} is {confusion_matrix_tl_ft}"
                quote_f1score = f"F1_score in {fold_no} is {f1score_ft}"

                # Open txt and put them in with access mode 'a'
                file_object_fine = open(f"FT_{basename}.txt", 'a')

                # Append the objects
                file_object_fine.write(quote_acc + '\n' + quote_testerror + '\n' + quote_loss+ '\n' +quote_confusionmatrix + '\n'+ quote_f1score+ '\n'+ '\n')
                file_object_fine.close()

            elif type == 'finetuning_all':

                # Renaming the label lists (verbatim - what they are)
                training_labels = change_labels_verbatim(labels_3D[train])
                test_labels = change_labels_verbatim(labels_3D[test])

                # Renaming the label lists
                new_labels_train = []
                for label in training_labels[0:]:
                    if label == 'Condition1':
                        newlabel = 0
                    else:
                        newlabel = 1
                    new_labels_train.append(newlabel)

                new_labels_test = []
                for label in test_labels[0:]:
                    if label == 'Condition1':
                        newlabel = 0
                    else:
                        newlabel = 1
                    new_labels_test.append(newlabel)

                train_dataset = tf.data.Dataset.from_tensor_slices((samples_normed[train], new_labels_train)) 
                test_dataset = tf.data.Dataset.from_tensor_slices((samples_normed[test], new_labels_test))

                BATCH_SIZE = 4

                SHUFFLE_BUFFER_SIZE = len(samples_normed[train]) 
                train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
                test_dataset = test_dataset.batch(BATCH_SIZE)

                history = model.fit(train_dataset,
                    epochs=20,
                    batch_size=4)

                model.save(f"model_FTA{participant}_Fold{fold_no}.h5")

                # Saving dataframe of train acc/loss
                hist_finetuning_all_df = pd.DataFrame(history.history)

                # Saving test metrics
                loss_fta, accuracy_fta = model.evaluate(test_dataset)
                y_pred = (model.predict(test_dataset).ravel()>0.5)+0
                confusion_matrix_tl_fta = confusion_matrix(new_labels_test,y_pred)
                f1score_fta = f1_score(new_labels_test, y_pred, average="weighted")

                # SCORES FOR LATER IN TXT FILE
                quote_acc = f"Test Accuracy in {fold_no} is {accuracy_fta}"
                quote_testerror = f"Test Error Rate in {fold_no} is {1-accuracy_fta}" 
                quote_loss = f"Test Loss in {fold_no} is {loss_fta} "
                quote_confusionmatrix = f"Confusion matrix in {fold_no} is {confusion_matrix_tl_fta}"
                quote_f1score = f"F1_score in {fold_no} is {f1score_fta}"

                # Open txt and put them in with access mode 'a'
                file_object_fine_a = open(f"FTA_{basename}.txt", 'a')

                # Append the objects
                file_object_fine_a.write(quote_acc + '\n' + quote_testerror + '\n' + quote_loss+ '\n' +quote_confusionmatrix + '\n'+ quote_f1score+ '\n'+ '\n')

                file_object_fine_a.close()

            elif type == 'from_scratch':
                # --- CREATE MODEL FROM SCRATCH FOR COMPARISON ---
                model_frombottom = get_model(width=36, height=36, depth=81)
                model_frombottom.summary()
                
                model_frombottom.compile(
                    loss="binary_crossentropy",
                    optimizer=keras.optimizers.Adam(0.00001),
                    metrics=["acc"],
                )

                # Renaming the label lists (verbatim - what they are)
                training_labels = change_labels_verbatim(labels_3D[train])
                test_labels = change_labels_verbatim(labels_3D[test])

                # Renaming the label lists (binary - manual solution)
                new_labels_train = []
                for label in training_labels[0:]:
                    if label == 'Condition1':
                        newlabel = 0
                    else:
                        newlabel = 1
                    new_labels_train.append(newlabel)

                new_labels_test = []
                for label in test_labels[0:]:
                    if label == 'Condition1':
                        newlabel = 0
                    else:
                        newlabel = 1
                    new_labels_test.append(newlabel)

                train_dataset = tf.data.Dataset.from_tensor_slices((samples_normed[train], new_labels_train)) 
                test_dataset = tf.data.Dataset.from_tensor_slices((samples_normed[test], new_labels_test))

                BATCH_SIZE = 4

                SHUFFLE_BUFFER_SIZE = len(samples_normed[train]) # shuffle randomly shuffles the elements. 
                train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
                test_dataset = test_dataset.batch(BATCH_SIZE)

                history = model_frombottom.fit(train_dataset,
                    epochs=20,
                    batch_size=4)

                model_frombottom.save(f"model_frombottom_{participant}_Fold{fold_no}.h5")

                # Saving dataframe of train acc/loss
                hist_model_scratch_df = pd.DataFrame(history.history)

                # Saving test metrics
                loss1, accuracy1 = model_frombottom.evaluate(test_dataset) # NOTE:
                y_pred_bottom = (model_frombottom.predict(test_dataset).ravel()>0.5)+0
                f1score1 =f1_score(new_labels_test, y_pred_bottom, average="weighted")
                confusion_matrix_bottom = confusion_matrix(new_labels_test,y_pred_bottom)

                # SCORES FOR LATER IN TXT FILE
                quote_acc = f"Test Accuracy in {fold_no} is {accuracy1}"
                quote_testerror = f"Test Error Rate in {fold_no} is {1-accuracy1}" 
                quote_loss = f"Test Loss in {fold_no} is {loss1} "
                quote_confusionmatrix = f"Confusion matrix in {fold_no} is {confusion_matrix_bottom}"
                quote_f1score = f"F1_score in {fold_no} is {f1score1}"

                # Open txt and put them in with access mode 'a'
                file_object_scratch = open(f"Scratch_{basename}.txt", 'a')

                # Append the objects
                file_object_scratch.write(quote_acc + '\n' + quote_testerror + '\n' + quote_loss+ '\n' +quote_confusionmatrix + '\n'+ quote_f1score+ '\n'+'\n')
                file_object_scratch.close()

                print(f'Participant done with fold nr. {fold_no} ')

                dataframes = [hist_feature_extraction_df, hist_finetuning_df,hist_finetuning_all_df,hist_model_scratch_df]
                bigdataframe_fold = pd.concat(dataframes)
                
                # Appending to global list of dataframes for this person
                participant_dataframes.append(bigdataframe_fold)
                fold_no = fold_no + 1

    participant_dataframes_concat = pd.concat(participant_dataframes) # For one person, this dataframe should include 5 x 3 chunks (5 folds of three different finetuning ways)  
    file_name = f"HISTORY_{basename}_allfolds.csv"
    participant_dataframes_concat.to_csv(file_name)
    print(f"DataFrame for {participant} for all folds are written to a csv File successfully.")












