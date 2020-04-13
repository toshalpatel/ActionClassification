from scipy import io as sio
from read_datasetBreakfast import load_data, read_mapping_dict, load_one_data
import os

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tf_models, utils

import numpy as np
import pdb
import pandas as pd
import scipy.stats as ss

COMP_PATH = ''

''' 
training to load train set
test to load test set
'''

train_split =  os.path.join(COMP_PATH, 'splits/train.split1.bundle') #Train Split
test_split  =  os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
GT_folder   =  os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video 
DATA_folder =  os.path.join(COMP_PATH, 'data/') #Frame I3D features for all videos
mapping_loc =  os.path.join(COMP_PATH, 'splits/mapping_bf.txt')
test_mapping_loc = os.path.join(COMP_PATH, 'segment.txt')

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices: tf.config.experimental.set_memory_growth(device, True)

#Get feautres and labels for train data and just features for test
actions_dict = read_mapping_dict(mapping_loc)

split = 'training'

if  split == 'training':
    data_feat, data_labels, data_labels_loc = load_data(train_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features and labels
if  split == 'test':
    data_feat = load_data(test_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features only


model_ = ["BaselineLinear","ED-TCN"][0]
print(model_)

## Testing and validation data

#one-hot encoding
modified_labels_list = []
for i in range(len(data_feat)):
    final = np.zeros((data_feat[i].shape[0], 48))
    label = data_labels[i]
    loc = data_labels_loc[i]
    for j in range(len(label)):
        final[loc[j]:loc[j+1], label[j]] = 1
    for i in range(len(final)):
        if(np.sum(final[i]) == 0):
            final[i, 0] = 1
    modified_labels_list.append(final)


X_train, X_test, y_train, y_test = train_test_split(data_feat, modified_labels_list, test_size=0.15, random_state=42)
train_lengths = [x.shape[0] for x in X_train]
test_lengths = [x.shape[0] for x in X_test]

nb_epoch = 200
n_classes = len(actions_dict)
n_feat = 400

## Initializing the model

if model_ == "BaselineLinear":

    max_len = max(np.max(train_lengths), np.max(test_lengths))
    print(max_len)

    # In order process batches simultaneously all data needs to be of the same length
    # So make all same length and mask out the ends of each.
    X_train_m, Y_train_, M_train = utils.mask_data(X_train, y_train, max_len, mask_value=0)
    X_test_m, Y_test_, M_test = utils.mask_data(X_test, y_test, max_len, mask_value=0)
    
   # pdb.set_trace()
    
    X_train_m = X_train_m.reshape(-1,n_feat)
    X_test_m = X_test_m.reshape(-1,n_feat)
    Y_train_ = Y_train_.reshape(-1,n_classes)
    Y_test_ = Y_test_.reshape(-1,n_classes)
    
    model = tf_models.BaselineLinear(n_feat, n_classes)
    
    
elif model_ == "ED-TCN":
    #ED-TCN parameters
    n_nodes = [64, 96]
    conv = 25
    causal = False

    n_train = len(X_train)
    n_test = len(X_test)
    
    n_layers = len(n_nodes)
    max_len = max(np.max(train_lengths), np.max(test_lengths))
    max_len = int(np.ceil(max_len / (2**n_layers)))*2**n_layers
    
    # In order process batches simultaneously all data needs to be of the same length
    # So make all same length and mask out the ends of each.
    X_train_m, Y_train_, M_train = utils.mask_data(X_train, y_train, max_len, mask_value=0)
    X_test_m, Y_test_, M_test = utils.mask_data(X_test, y_test, max_len, mask_value=0)
    
    model, param_str = tf_models.ED_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal, \
                                activation='norm_relu', return_param_str=True)

    

## TRAINING

if split == 'training':

    if model_ == "BaselineLinear":

        checkpoint = ModelCheckpoint('results/model_baslineLinear-{epoch:03d}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')  
        pdb.set_trace()
        model.fit(X_train_m, Y_train_, nb_epoch=nb_epoch, batch_size=8, validation_data=(X_test_m, Y_test_),verbose=1,callbacks=[checkpoint]) 


    elif model_ == "ED-TCN":

        checkpoint = ModelCheckpoint('results/model_edtcn-{epoch:03d}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')  
        model.fit(X_train_m, Y_train_, nb_epoch=nb_epoch, batch_size=8, validation_data=(X_test_m, Y_test_),
                    verbose=1, sample_weight=M_train[:,:,0], callbacks=[checkpoint]) 

    ## save model
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_+"_model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_+"_model.h5")
    print("Model saved")

    #testing on validation data
    split = 'test'


## TEST

elif split == 'test':

    model.load_weights('results/'+model_+'.h5')
    output = model.predict(X_test_m, verbose=0)
    
    xx = np.argmax(output, axis=2)
    print(xx[0])
    yy = np.argmax(Y_test_, axis=2)
    print(yy[0])

    locations = []
    for i in range(yy.shape[0]):
        out = [0]
        last = yy[i, 0]
        for j in range(1, yy.shape[1]):
            if(yy[i, j] != last):
                last = yy[i, j]
                out.append(j)
        locations.append(out)
    print(locations[-1])

    # checking over-segmentation of predicted labels
    locations_p=[]
    for i in range(xx.shape[0]):
        out = [0]
        last = xx[i, 0]
        for j in range(1, xx.shape[1]):
            if(xx[i, j] != last):
                last = xx[i, j]
                out.append(j)
        locations_p.append(out)
    print(locations_p[-1])


    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            if(xx[i, j] == None):
                xx[i, j] = -1


    for i in range(yy.shape[0]):
        for j in range(yy.shape[1]):
            if(yy[i, j] == None):
                yy[i, j] = -1


    labels = []
    for i in range(xx.shape[0]):
        l = []
        for j in range(1,len(locations[i]) - 1):
            l.append(int(ss.mode(xx[i,locations[i][j]:locations[i][j+1]]).mode))
        labels.append(l)

    not_same_labels=[]
    for i in range(xx.shape[0]):
        l = [0]
        for j in range(1,len(locations[i]) - 1):
            seg = xx[i,locations[i][j]:locations[i][j+1]]
            out=[]
            for f in range(len(seg)):
                if seg[f] == l[j-1]: out.append(-1)
                else: out.append(seg[f])
            if -1 in out: out.remove(-1)
            l.append(int(ss.mode(out).mode))
        del l[0]
        not_same_labels.append(l)

    prob_labels = []
    for i in range(output.shape[0]):
        l = []
        for j in range(1,len(locations[i]) - 1):
            l.append(np.argmax(np.sum(output[i, locations[i][j]:locations[i][j+1], :], axis=0)))
        prob_labels.append(l)


    true_labels = []
    for i in range(yy.shape[0]):
        l = []
        for j in range(1,len(locations[i]) - 1):
            l.append(int(ss.mode(yy[i,locations[i][j]:locations[i][j+1]]).mode))
        true_labels.append(l)


    pred_labels=[]
    gt_labels=[]
    pprob_labels=[]
    ns_labels=[]

    for i in labels:
        for j in i:
            pred_labels.append(j)

    for i in prob_labels:
        for j in i:
            pprob_labels.append(j)

    for i in true_labels:
        for j in i:
            gt_labels.append(j)

    for i in not_same_labels:
        for j in i:
            ns_labels.append(j)

    print(len(pred_labels))
    print(len(gt_labels))
    print(len(pprob_labels))
    print(len(ns_labels))

    print("pred_labels acc:")
    print(accuracy_score(gt_labels, pred_labels))
    print("prob_labels acc:")
    print(accuracy_score(gt_labels, pprob_labels))
    print("ns_labels acc:")
    print(accuracy_score(gt_labels, ns_labels))

