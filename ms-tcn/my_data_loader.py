import torch
import numpy as np
import random


class MyDataLoader(object):
    def __init__(self, actions_dict, data_breakfast, labels_breakfast, test_ratio):
        list_of_examples = list(range(len(data_breakfast)))
        random.shuffle(list_of_examples)

        self.test_list_len = int(test_ratio * len(list_of_examples))
        self.test_list = list_of_examples[:self.test_list_len]
        self.validation_list = list_of_examples[self.test_list_len:]

        self.current_index = 0
        self.current_validation_index = 0
        self.num_classes = len(actions_dict)
        self.actions_dict = actions_dict

        self.data_breakfast = data_breakfast
        self.labels_breakfast = labels_breakfast

    def reset(self):
        self.current_index = 0
        self.current_validation_index = 0
        random.shuffle(self.test_list)

    def has_next_test(self):
        if self.current_index < len(self.test_list):
            return True
        return False

    def next_test_batch(self, batch_size):
        batch_indexes = self.test_list[self.current_index:self.current_index + batch_size]
        self.current_index += batch_size

        batch_inputs_list = []
        batch_target_list = []
        for index in batch_indexes:
            batch_inputs_list.append(self.data_breakfast[index].transpose(1, 0))
            batch_target_list.append(self.labels_breakfast[index])

        max_length_of_sequences = max(map(len, batch_target_list))

        batch_inputs_tensor = torch.zeros(len(batch_inputs_list), batch_inputs_list[0].shape[0], max_length_of_sequences, dtype=torch.float)
        batch_targets_tensor = torch.ones(len(batch_inputs_list), max_length_of_sequences, dtype=torch.long) * (-100)
        mask = torch.zeros(batch_inputs_tensor.shape, dtype=torch.float)
        for i in range(len(batch_inputs_list)):
            batch_inputs_tensor[i, :, :batch_inputs_list[i].shape[1]] = batch_inputs_list[i]
            batch_targets_tensor[i, :batch_target_list[i].shape[0]] = batch_target_list[i]
            mask[i, :, :np.shape(batch_target_list[i])[0]] = torch.ones(batch_inputs_list[i].shape)

        batch_inputs_tensor_with_mask = torch.stack([batch_inputs_tensor, mask], dim=0)

        return batch_inputs_tensor_with_mask, batch_targets_tensor

    def has_next_validation(self):
        if self.current_validation_index < len(self.validation_list):
            return True
        return False

    def next_validation(self):
        index = self.validation_list[self.current_validation_index]
        self.current_validation_index += 1

        inputs = self.data_breakfast[index].transpose(1, 0).unsqueeze(0).float()
        targets = self.labels_breakfast[index].unsqueeze(0)
        mask = torch.ones(inputs.shape, dtype=torch.float)

        inputs_with_mask = torch.stack([inputs, mask], dim=0)
        return inputs_with_mask, targets
