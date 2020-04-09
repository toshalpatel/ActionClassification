import torch
import torch.nn as nn


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_in = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x, mask = x
        out = self.conv_in(x)
        out = self.relu(out)
        out = self.conv_out(out)
        out = self.dropout(out)
        out = (x + out) * mask
        out = torch.stack([out, mask], dim=0)
        return out


class SingleTCN(nn.Module):
    def __init__(self, num_layers_pre_stage, num_features_per_layer, input_features_dim, output_feature_dim):
        super(SingleTCN, self).__init__()
        self.num_features_per_layer = num_features_per_layer
        self.output_feature_dim = output_feature_dim
        self.conv_in = nn.Conv1d(input_features_dim, num_features_per_layer, 1)

        layers = []
        for i in range(num_layers_pre_stage):
            dilation_size = 2 ** i
            layers.append(DilatedResidualLayer(dilation_size, num_features_per_layer, num_features_per_layer))
        self.network = nn.Sequential(*layers)

        self.conv_out = nn.Conv1d(num_features_per_layer, output_feature_dim, 1)

    def forward(self, x):
        x, mask = x
        out = self.conv_in(x)
        new_mask = torch.stack([mask[:, 0, :]] * self.num_features_per_layer, dim=1)
        out = torch.stack([out, new_mask], dim=0)

        out = self.network(out)

        new_mask = torch.stack([mask[:, 0, :]] * self.output_feature_dim, dim=1)
        out = self.conv_out(out[0]) * new_mask

        out = torch.stack([out, new_mask], dim=0)
        return out


class MultiStageTCN(nn.Module):
    def __init__(self, num_stages, num_layers_pre_stage, num_features_per_layer, input_features_dim, output_feature_dim):
        super(MultiStageTCN, self).__init__()
        self.stage1 = SingleTCN(num_layers_pre_stage, num_features_per_layer, input_features_dim, output_feature_dim)
        self.softmax = nn.Softmax(dim=1)

        self.stages = nn.ModuleList(
            [SingleTCN(num_layers_pre_stage, num_features_per_layer, output_feature_dim, output_feature_dim) for s in
             range(num_stages - 1)])

    def forward(self, x):
        out, mask = self.stage1(x)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = self.softmax(out) * mask
            out = torch.stack([out, mask], dim=0)
            out = s(out)[0]
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs, mask
