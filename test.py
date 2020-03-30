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
X_train, data_labels, data_labels_loc = load_data(train_split, actions_dict, GT_folder, DATA_folder, datatype = 'training')

X_test = load_data(test_split, actions_dict, GT_folder, DATA_folder, datatype = 'test')

modified_labels_list = []
for i in range(len(X_train)):
    final = np.zeros((X_train[i].shape[0], 48))
    label = data_labels[i]
    loc = data_labels_loc[i]
    for j in range(len(label)):
        final[loc[j]:loc[j+1], label[j]] = 1
    for i in range(len(final)):
        if(np.sum(final[i]) == 0):
            final[i, 0] = 1
    modified_labels_list.append(final)

y_train = modified_labels_list

print("data loaded successfully!")

n_nodes = [64, 96]
nb_epoch = 50
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
n_layers = len(n_nodes)
max_len = max(np.max(train_lengths), np.max(test_lengths))
max_len = int(np.ceil(max_len / (2**n_layers)))*2**n_layers
print("Max length:", max_len)

# pdb.set_trace()

X_train_m, Y_train_, M_train = utils.mask_data(X_train, y_train, max_len, mask_value=0)
X_test_m, M_test = utils.mask_test_data(X_test, max_len, mask_value=0)

model, param_str = tf_models.ED_TCN(n_nodes, conv, n_classes, n_feat, max_len, causal=causal, 
                            activation='norm_relu', return_param_str=True)

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
# model_lstm, param_str = tf_models.BidirLSTM(n_nodes[0], n_classes, n_feat, causal=causal, return_param_str=True)


checkpoint = ModelCheckpoint('results/model_test_edtcn-{epoch:03d}.h5', verbose=1, monitor='loss', save_best_only=True, mode='auto')  

# model.load_weights('results/model_secondTraining-038.h5')

# TRAINING
model.fit(X_train_m, Y_train_, nb_epoch=nb_epoch, batch_size=8,
            verbose=1, sample_weight=M_train[:,:,0], callbacks=[checkpoint]) 


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model")


## TEST
output = model.predict(X_test_m, verbose=0)
xx = np.argmax(output, axis=2)

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        if(xx[i, j] == None):
            xx[i, j] = -1

## load test segments
with open(test_mapping_loc) as f:
    lines = f.read().splitlines()
test_segments=[x.split(" ") for x in lines]

test_labels_loc = []
for i in test_segments:
    out=[]
    for j in i:
        out.append(int(j))
    test_labels_loc.append(out)
    

## majority voting
ans_labels=[]
l=[]
for i in range(xx.shape[0]):
    x=[]
    for j in range(1,len(test_labels_loc[i]) - 1):
        l.append(int(ss.mode(xx[i,test_labels_loc[i][j]:test_labels_loc[i][j+1]]).mode))
        x.append(int(ss.mode(xx[i,test_labels_loc[i][j]:test_labels_loc[i][j+1]]).mode))
    ans_labels.append(x)
    

segment_labels=pd.DataFrame()
segment_labels['segment_labels']=ans_labels
segment_labels.to_csv("segment_labels.csv", index=None)
    
predictions=pd.DataFrame()
predictions['Category'] = l
predictions['Id'] = predictions.index
predictions = predictions[['Id', 'Category']]
predictions.to_csv("predictions.csv",index=None)