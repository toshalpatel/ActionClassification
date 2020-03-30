from scipy import io as sio
import os
from keras.utils import np_utils
import tensorflow as tf
from read_datasetBreakfast import load_data, read_mapping_dict, load_one_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# TCN imports 
import tf_models, datasets, utils, metrics
from utils import imshow_
import numpy as np
from keras.callbacks import ModelCheckpoint
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
data_feat, data_labels, data_labels_loc = load_data(train_split, actions_dict, GT_folder, DATA_folder, datatype = 'training')
#data_feat, data_labels = load_data( train_split, actions_dict, GT_folder, DATA_folder, datatype = 'training')
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

# actions_dict = read_mapping_dict(mapping_loc)
# if  split == 'training':
#     data_feat, data_labels = load_data( train_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features and labels
# if  split == 'test':
#     data_feat = load_data( test_split, actions_dict, GT_folder, DATA_folder, datatype = split) #Get features only

print("data loaded successfully!")


n_nodes = [64, 96]
nb_epoch = 200
video_rate = 3

conv = 25

# In order process batches simultaneously all data needs to be of the same length
# So make all same length and mask out the ends of each.

causal = False
n_classes = len(actions_dict)

train_lengths = [x.shape[0] for x in X_train]
test_lengths = [x.shape[0] for x in X_test]
n_train = len(X_train)
n_test = len(X_test)

n_feat = 400
# n_feat = data.n_features
# print("# Feat:", n_feat)
n_layers = len(n_nodes)
max_len = max(np.max(train_lengths), np.max(test_lengths))
max_len = int(np.ceil(max_len / (2**n_layers)))*2**n_layers
print("Max length:", max_len)

X_train_m, Y_train_, M_train = utils.mask_data(X_train, y_train, max_len, mask_value=0)
X_test_m, Y_test_, M_test = utils.mask_data(X_test, y_test, max_len, mask_value=0)


# X_test_m = X_test_m.reshape(-1,1,400) #np.reshape(X_test_m, [-1, 1, 400])
# Y_test_ = np.reshape(Y_test_, [-1, 1, 48])
# # x = X_test_m
# # y = Y_test_
# # for i in range(8039): 
# #     x = np.concatenate([x, X_test_m], axis=1)
# #     y = np.concatenate([y, Y_test_], axis=1)
    
# # X_test_m = x
# # Y_test_ = y

model, param_str = tf_models.ED_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal, 
                            activation='norm_relu', return_param_str=True)

# if model_type == "tCNN":
#     model, param_str = tf_models.temporal_convs_linear(n_nodes[0], conv, n_classes, n_feat, 
#                                         max_len, causal=causal, return_param_str=True)
# elif model_type == "ED-TCN":
#     model, param_str = tf_models.ED_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal, 
#                             activation='norm_relu', return_param_str=True) 
#     # model, param_str = tf_models.ED_TCN_atrous(n_nodes, conv, n_classes, n_feat, max_len, 
#                         # causal=causal, activation='norm_relu', return_param_str=True)                 
# elif model_type == "TDNN":
#     model, param_str = tf_models.TimeDelayNeuralNetwork(n_nodes, conv, n_classes, n_feat, max_len, 
#                        causal=causal, activation='tanh', return_param_str=True)
# elif model_type == "DilatedTCN":
#     model, param_str = tf_models.Dilated_TCN(n_feat, n_classes, n_nodes[0], L, B, max_len=max_len, 
#                             causal=causal, return_param_str=True)
# elif model_type == "LSTM":
#     model, param_str = tf_models.BidirLSTM(n_nodes[0], n_classes, n_feat, causal=causal, return_param_str=True)


checkpoint = ModelCheckpoint('results/model_tcn25-{epoch:03d}.h5', verbose=1, monitor='val_loss', save_best_only=True, mode='auto')  

# model.load_weights('results/model_secondTraining-038.h5')

# TRAINING
model.fit(X_train_m, Y_train_, nb_epoch=nb_epoch, batch_size=8, validation_data=(X_test_m, Y_test_),
            verbose=1, sample_weight=M_train[:,:,0], callbacks=[checkpoint]) 


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model")


# # TEST

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

print(true_labels[-1])
print(prob_labels[-1])
print(labels[-1])

pred_labels=[]
gt_labels=[]
pprob_labels=[]

for i in labels:
    for j in i:
        pred_labels.append(j)
        
for i in prob_labels:
    for j in i:
        pprob_labels.append(j)
        
for i in true_labels:
    for j in i:
        gt_labels.append(j)
        
print(len(pred_labels))
print(len(gt_labels))
print(len(pprob_labels))

print("pred_labels acc:")
print(accuracy_score(gt_labels, pred_labels))
print("prob_labels acc:")
print(accuracy_score(gt_labels, pprob_labels))



trial_metrics = metrics.ComputeMetrics(overlap=.1, bg_class=0)
AP_test = model.predict(X_test_m, verbose=0)
AP_test = utils.unmask(AP_test, M_test)
P_test = [p.argmax(1) for p in AP_test])
trial_metrics.add_predictions(split, P_test, y_test)       
trial_metrics.print_trials()
trial_metrics.print_scores()
trial_metrics.print_trials()
print()


# output = model.predict(X_test_m, verbose=0)


# locations = []
# yy = np.argmax(Y_test_, axis=2)
# for i in range(yy.shape[0]):
#     out = [0]
#     last = yy[i, 0]
#     for j in range(1, yy.shape[1]):
#         if(yy[i, j] != last):
#             last = yy[i, j]
#             out.append(j)
#     locations.append(out)

# xx = np.argmax(output, axis=2)


# for i in range(xx.shape[0]):
#     for j in range(xx.shape[1]):
#         if(xx[i, j] == None):
#             xx[i, j] = -1

        
# for i in range(yy.shape[0]):
#     for j in range(yy.shape[1]):
#         if(yy[i, j] == None):
#             yy[i, j] = -1

        
# labels = []
# for i in range(xx.shape[0]):
#     l = []
#     for j in range(1,len(locations[i]) - 1):
#         l.append(int(ss.mode(xx[i,locations[i][j]:locations[i][j+1]]).mode))
#     labels.append(l)
        

# prob_labels = []
# for i in range(output.shape[0]):
#     l = []
#     for j in range(1,len(locations[i]) - 1):
#         l.append(np.argmax(np.sum(output[i, locations[i][j]:locations[i][j+1], :], axis=0)))
#     prob_labels.append(l)


# true_labels = []
# for i in range(yy.shape[0]):
#     l = []
#     for j in range(1,len(locations[i]) - 1):
#         l.append(int(ss.mode(yy[i,locations[i][j]:locations[i][j+1]]).mode))
#     true_labels.append(l)

