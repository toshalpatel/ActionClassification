import time

import torch
from torch import nn, optim
import torch.nn.functional as F

from model import MultiStageTCN
from my_data_loader import MyDataLoader
import os
import argparse
import random

from read_datasetBreakfast import read_mapping_dict, load_one_data, load_test_segments, load_data

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 5242
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
args = parser.parse_args()

COMP_PATH = ''

train_split = os.path.join(COMP_PATH, 'splits/train.split1.bundle') #Train Split
test_split = os.path.join(COMP_PATH, 'splits/test.split1.bundle') #Test Split
GT_folder = os.path.join(COMP_PATH, 'groundTruth/') #Ground Truth Labels for each training video
DATA_folder = os.path.join(COMP_PATH, 'data/') #Frame I3D features for all videos
mapping_loc = os.path.join(COMP_PATH, 'splits/mapping_bf.txt')
model_folder = os.path.join(COMP_PATH, './models/')
test_segment_loc = os.path.join(COMP_PATH, './test_segment.txt')
predict_result_loc = os.path.join(COMP_PATH, './ans.csv')
record_file_loc = os.path.join(COMP_PATH, './record.txt')


actions_dict = read_mapping_dict(mapping_loc)

if not os.path.exists(model_folder):
    os.makedirs(model_folder)

output_feature_dim = len(actions_dict)

num_stages = 4
num_layers_per_stage = 12
num_features_per_layer = 64 #number of features that are extracted
input_features_dim = 400
batch_size = 16 
#batch size =1 is not good as during backprop, if each training example is considered, it will affect the gradient a lot. Hence, if we consider more number of training examples, higher batch size is better 
lr = 0.0005
training_epochs = 100
start_epoch = 0
predict_epoch = 75
test_ratio = 0.95
MSE_loss_factor = 0.20

model = MultiStageTCN(num_stages, num_layers_per_stage, num_features_per_layer, input_features_dim, output_feature_dim)
ce = nn.CrossEntropyLoss(ignore_index=-100)
mse = nn.MSELoss(reduction='none')


def train():
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if start_epoch != 0:
        model.load_state_dict(torch.load(model_folder + "/epoch-%d.model" % start_epoch))
        optimizer.load_state_dict(torch.load(model_folder + "/epoch-%d.opt" % start_epoch))
    start_time = time.time()
    f = open(record_file_loc, 'a')
    for epoch in range(start_epoch + 1, training_epochs + start_epoch + 1):
        model.train()
        epoch_loss, correct, total, batch_num = 0, 0, 0, 0

        while data_loader.has_next_test():
            batch_num += 1
            if batch_num % 10 == 0:
                print("--- %s seconds ---" % (time.time() - start_time))
                print("batch_number = %d, loss = %f, acc = %f" % (batch_num, epoch_loss / batch_num, correct / total))
            batch_inputs_tensor_with_mask, batch_target = data_loader.next_test_batch(batch_size)
            batch_inputs_tensor_with_mask, batch_target = batch_inputs_tensor_with_mask.to(device), batch_target.to(device)
            optimizer.zero_grad()
            predictions, mask = model(batch_inputs_tensor_with_mask)

            loss = 0
            for p in predictions:
                loss += ce(p.transpose(2, 1).contiguous().view(-1, output_feature_dim), batch_target.view(-1))
                loss += MSE_loss_factor * torch.mean(torch.clamp(mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16) * mask[:, 0:output_feature_dim, 1:])

            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(predictions[-1].data, 1)
            correct += ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1)).sum().item()
            total += torch.sum(mask[:, 0, :]).item()

        torch.save(model.state_dict(), model_folder + "epoch-" + str(epoch) + ".model")
        torch.save(optimizer.state_dict(), model_folder + "epoch-" + str(epoch) + ".opt")

        validation_loss, validation_acc = validate()
        info = "[epoch %d]: train_loss = %f, train_acc = %f, validation_loss = %f, validation_acc = %f" \
               % (epoch, epoch_loss / data_loader.test_list_len, correct / total, validation_loss, validation_acc)

        data_loader.reset()
        print(info)
        f.write(info)
        f.write("\n")
        f.flush()
    f.close()


def validate():
    model.eval()
    epoch_loss, correct, total, batch_num = 0, 0, 0, 0
    while data_loader.has_next_validation():
        with torch.no_grad():
            inputs_with_mask, targets = data_loader.next_validation()
            inputs_with_mask, targets = inputs_with_mask.float().to(device), targets.long().to(device)
            predictions, mask = model(inputs_with_mask)

            loss = 0
            for p in predictions:
                loss += ce(p.transpose(2, 1).contiguous().view(-1, output_feature_dim), targets.view(-1))
                loss += MSE_loss_factor * torch.mean(torch.clamp(mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16) * mask[:, :output_feature_dim, 1:])

            epoch_loss += loss.item()

            _, predicted = torch.max(predictions[-1].data, 1)
            correct += ((predicted == targets).float() * mask[:, 0, :].squeeze(1)).sum().item()
            total += torch.sum(mask[:, 0, :]).item()

    return epoch_loss / len(data_loader.validation_list), correct / total


def predict():
    model.eval()
    ans = []
    number = -1
    with torch.no_grad():
        model.to(device)
        model.load_state_dict(torch.load(model_folder + "epoch-%d.model" % predict_epoch))

        for data in data_breakfast:
            number += 1
            data = data.transpose(1, 0).float()
            data.unsqueeze_(0)
            data = torch.stack([data, torch.ones(data.shape)], dim=0)
            data = data.to(device)
            predictions, mask = model(data)
            _, predicted = torch.max(predictions[-1].data, 1)
            predicted = predicted.squeeze()          
            
            for i in range(len(segments[number]) - 1):
                start = int(segments[number][i])
                end = int(segments[number][i+1])
                segment = {}
                for j in range(start, end):
                    prediction = predicted[j].item()
                    if prediction not in segment and prediction != 0:
                        segment[prediction] = 1
                    elif prediction != 0:
                        segment[prediction] += 1
                action_num = 0
                action = 0
                for prediction in segment:
                    if segment[prediction] > action_num:
                        action_num = segment[prediction]
                        action = prediction
                ans.append(action)

    print("Total Segment： %d" % len(ans))
    with open(predict_result_loc, "w") as f:
        f.write("Id,Category\n")

        for i in range(len(ans)):
            f.write(str(i) + "," + str(ans[i]) + "\n")


if args.action == "train":
    data_breakfast, labels_breakfast = load_data(train_split, actions_dict, GT_folder, DATA_folder, datatype='training')
    data_loader = MyDataLoader(actions_dict, data_breakfast, labels_breakfast, test_ratio)
    train()

if args.action == "predict":
    data_breakfast = load_data(test_split, actions_dict, GT_folder, DATA_folder, datatype='test')
    segments = load_test_segments(test_segment_loc)
    predict()
