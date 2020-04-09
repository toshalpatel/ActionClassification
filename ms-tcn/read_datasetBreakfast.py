import os
import torch
import numpy as np
import os.path 
 
 
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def load_one_data(split_load, actions_dict, GT_folder, DATA_folder, datatype='training'):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[1:-1][0:20]
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]

    if datatype == 'training':
        data_breakfast = []
        labels_breakfast = []
        for content in content_all:

            file_ptr = open(GT_folder + content, 'r')
            curr_gt = file_ptr.read().split('\n')[:-1]

            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'

            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            label_curr_video = []
            for iik in range(len(curr_gt)):
                label_curr_video.append(actions_dict[curr_gt[iik]])

            data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))
            labels_breakfast.append(torch.tensor(label_curr_video, dtype=torch.float64))

        # labels_uniq, labels_uniq_loc = get_label_bounds(labels_breakfast)
        print("Finish Load the Training data and labels!!!")
        return data_breakfast, labels_breakfast
    if datatype == 'test':
        data_breakfast = []
        for content in content_all:
            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'

            curr_data = np.loadtxt(loc_curr_data, dtype='float32')

            data_breakfast.append(torch.tensor(curr_data, dtype=torch.float64))

        print("Finish Load the Test data!!!")
        return data_breakfast


def load_data(split_load, actions_dict, GT_folder, DATA_folder, datatype = 'training',):
    file_ptr = open(split_load, 'r')
    content_all = file_ptr.read().split('\n')[1:-1]
    content_all = [x.strip('./data/groundTruth/') + 't' for x in content_all]

    if datatype == 'training':
        data_breakfast = []
        labels_breakfast = []
        for content in content_all:
        
            file_ptr = open( GT_folder + content, 'r')
            curr_gt = file_ptr.read().split('\n')[:-1]

            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'
        
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            label_curr_video = []
            for iik in range(len(curr_gt)):
                label_curr_video.append( actions_dict[curr_gt[iik]] )
         
            data_breakfast.append(torch.tensor(curr_data,  dtype=torch.float64 ) )
            labels_breakfast.append(torch.tensor(label_curr_video, dtype=torch.float64))
    
        # labels_uniq, labels_uniq_loc = get_label_bounds(labels_breakfast)
        print("Finish Load the Training data and labels!!!")     
        return data_breakfast, labels_breakfast
    if datatype == 'test':
        data_breakfast = []
        for content in content_all:
        
            loc_curr_data = DATA_folder + os.path.splitext(content)[0] + '.gz'
        
            curr_data = np.loadtxt(loc_curr_data, dtype='float32')
            
            data_breakfast.append(torch.tensor(curr_data,  dtype=torch.float64 ) )
    
        print("Finish Load the Test data!!!")
        return data_breakfast


def get_label_bounds(data_labels):
    labels_uniq = []
    labels_uniq_loc = []
    for kki in range(0, len(data_labels) ):
        uniq_group, indc_group = get_label_length_seq(data_labels[kki])
        labels_uniq.append(uniq_group[1:-1])
        labels_uniq_loc.append(indc_group[1:-1])
    return labels_uniq, labels_uniq_loc


def get_label_length_seq(content):
    label_seq = []
    length_seq = []
    start = 0
    length_seq.append(0)
    for i in range(len(content)):
        if content[i] != content[start]:
            label_seq.append(content[start])
            length_seq.append(i)
            start = i
    label_seq.append(content[start])
    length_seq.append(len(content))

    return label_seq, length_seq


def read_mapping_dict(mapping_file):
    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]

    actions_dict=dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict


def load_test_segments(test_segment_loc):
    file_ptr = open(test_segment_loc, 'r')
    lines = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    segments = []
    for line in lines:
        segment = line.split(' ')
        segments.append(segment)

    return segments
        
 


