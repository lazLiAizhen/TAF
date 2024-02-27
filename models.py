import numpy as np
import torch
import torch.nn as nn
class MlpNet(nn.Module):
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),

                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                    nn.BatchNorm1d(num_features=layer_sizes[l_id + 1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# class DAGCCA

from models import MlpNet
import torch
import torch.nn as nn
from loss_utility import *
import torch.nn.functional as F

# class DeepGCCA(nn.Module):
#     def __init__(self, layer_sizes_list, input_size_list, outdim_size, use_all_singular_values=False, device=torch.device('gpu')):
#         super(DeepGCCA, self).__init__()
#         self.model_list = []
#         for i in range(len(layer_sizes_list)):
#             self.model_list.append(MlpNet(layer_sizes_list[i], input_size_list[i]).double())
#         self.model_list = nn.ModuleList(self.model_list)
#         self.gcca_loss = GCCA_loss
#
#     def forward(self, x_list):
#         """
#
#         x_%  are the vectors needs to be make correlated
#         dim = [batch_size, features]
#
#         """
#         # feature * batch_size
#         output_list = []
#         for x, model in zip(x_list, self.model_list):
#             output_list.append(model(x))
#
#         return output_list
class FeatureExtractor(nn.Module):
    def __init__(self, num_inputs, embed_size=256):
        super(FeatureExtractor, self).__init__()
        self.in_features = embed_size
        self.feature_layers = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, self.in_features),
            nn.ReLU())

        self.feature_decoder = nn.Sequential(  ###########
            nn.Linear(self.in_features, 512),  # 512   1136
            nn.ReLU(),
            nn.Linear(512, num_inputs)  # 512   1136
        )

    def output_num(self):
        return self.in_features

    def forward(self, x, is_dec=False):
        enc = self.feature_layers(x)
        dec = self.feature_decoder(enc)  ############
        if is_dec:
            return enc, dec
        else:
            return enc

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list


    def output_num(self):
        return self.in_features

    def forward(self, x, is_dec=False):
        enc = self.feature_layers(x)
        dec = self.feature_decoder(enc)  ############
        if is_dec:
            return enc, dec
        else:
            return enc

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list



class DAsonNet1(nn.Module):  # 域
    def __init__(self, in_dim, out_dim):
        super(DAsonNet1, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def output_num(self):
        return self.out_dim

    def forward(self, x):
        y = self.fc(x)
        return y

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list

class DAsonNet2(nn.Module):  # 域
    def __init__(self, in_dim, out_dim):
        super(DAsonNet2, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def output_num(self):
        return self.out_dim

    def forward(self, x):
        y = self.fc(x)
        return y

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list

class DAsonNet3(nn.Module):  # 域
    def __init__(self, in_dim, out_dim):
        super(DAsonNet3, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def output_num(self):
        return self.out_dim

    def forward(self, x):
        y = self.fc(x)
        return y

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list

class LabelPredictor1(nn.Module):  # 标签预测器
    def __init__(self, in_dim, out_dim):
        super(LabelPredictor1, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y = self.fc(x)
        return y

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list

class LabelPredictor2(nn.Module):  # 标签预测器
    def __init__(self, in_dim, out_dim):
        super(LabelPredictor2, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y = self.fc(x)
        return y

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list

class LabelPredictor3(nn.Module):  # 标签预测器
    def __init__(self, in_dim, out_dim):
        super(LabelPredictor3, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y = self.fc(x)
        return y

    def get_parameters(self):
        parameter_list = [{"params": self.parameters(), "lr_mult": 1, 'decay_mult': 2}]
        return parameter_list

class scDGCCA(nn.Module):

    def __init__(self,layer_sizes_list,input_size_list, num_classes=5):#input_dim=2000,
        super(scDGCCA, self).__init__()
        # self.sharedNet = nn.Linear(input_dim, 256)
        #
        # layer_sizes_list
        self.mlp1 = MlpNet(layer_sizes_list[0],input_size_list[0])
        self.mlp2 = MlpNet(layer_sizes_list[1], input_size_list[1])
        self.mlp3 = MlpNet(layer_sizes_list[2], input_size_list[2])
        self.mlp4 = MlpNet(layer_sizes_list[3], input_size_list[3])

        self.sonnet1 = nn.Linear(128, 32)#256, 32
        self.sonnet2 = nn.Linear(128, 32)#256, 32
        self.sonnet3 = nn.Linear(128, 32)#56, 32
        self.cls_fc_son1 = nn.Linear(32, num_classes)
        self.cls_fc_son2 = nn.Linear(32, num_classes)
        self.cls_fc_son3 = nn.Linear(32, num_classes)
        # print(self.mlp1,self.mlp2,self.mlp3,self.mlp4)
    def forward(self, data_src, data_tgt=0, label_src=0, mark=1):#data_tgt=0
        mmd_loss = 0
        if self.training == True:
            # data_src = self.mlp1(data_src)
            data_tgt = self.mlp4(data_tgt)
            data_tgt_son1 = self.sonnet1(data_tgt)
            pred_tgt_son1 = self.cls_fc_son1(data_tgt_son1)
            data_tgt_son2 = self.sonnet2(data_tgt)
            pred_tgt_son2 = self.cls_fc_son2(data_tgt_son2)
            data_tgt_son3 = self.sonnet3(data_tgt)
            pred_tgt_son3 = self.cls_fc_son3(data_tgt_son3)

            if mark == 1:
                data_src = self.mlp1(data_src)
                data_src = self.sonnet1(data_src)   #source1 features
                # mmd_loss += mmd.mmd(data_src, data_tgt_son1)
                mmd_loss = mmd(data_src, data_tgt_son1) #mmd1
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1) #1-2  1-3
                                               - torch.nn.functional.softmax(data_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son3, dim=1)))
                pred_src = self.cls_fc_son1(data_src)
                # t=F.log_softmax(pred_src, dim=1)###
                # print(t)####
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                # cls_loss = nn.CrossEntropyLoss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss / 2,data_src,data_tgt_son1

            if mark == 2:
                data_src = self.mlp2(data_src)#
                data_src = self.sonnet2(data_src)
                mmd_loss += mmd(data_src, data_tgt_son2)
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1)
                                               - torch.nn.functional.softmax(data_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son2, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son3, dim=1)))
                pred_src = self.cls_fc_son2(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss / 2,data_src

            if mark == 3:
                data_src = self.mlp3(data_src)#
                data_src = self.sonnet3(data_src)
                mmd_loss += mmd(data_src, data_tgt_son3)
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1)
                                               - torch.nn.functional.softmax(data_tgt_son1, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son3, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son2, dim=1)))
                pred_src = self.cls_fc_son3(data_src)
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss / 2,data_src

        else:
            data = self.mlp4(data_src)
            fea_son1 = self.sonnet1(data)
            pred1 = self.cls_fc_son1(fea_son1)

            fea_son2 = self.sonnet2(data)
            pred2 = self.cls_fc_son2(fea_son2)

            fea_son3 = self.sonnet3(data)
            pred3 = self.cls_fc_son3(fea_son3)

            return pred1, pred2, pred3, fea_son1


class scMDR(nn.Module):
    def __init__(self,layer_sizes_list,input_size_list, num_classes=5):
        super(scMDR, self).__init__()
        self.mlp1 = MlpNet(layer_sizes_list[0], input_size_list[0])
        self.mlp2 = MlpNet(layer_sizes_list[1], input_size_list[1])
        self.mlp3 = MlpNet(layer_sizes_list[2], input_size_list[2])
        self.mlp4 = MlpNet(layer_sizes_list[3], input_size_list[3])
        self.cls_fc1 = nn.Linear(128, num_classes)
        self.cls_fc2 = nn.Linear(128, num_classes)
        self.cls_fc3 = nn.Linear(128, num_classes)

    def forward(self, data_src, data_tgt=0, label_src=0, mark=1):  # data_tgt=0
        mmd_loss = 0
        if self.training == True:
            # data_src = self.mlp1(data_src)
            data_tgt = self.mlp4(data_tgt)
            data_tgt_son1 = self.sonnet1(data_tgt)
            pred_tgt_son1 = self.cls_fc_son1(data_tgt_son1)
            data_tgt_son2 = self.sonnet2(data_tgt)
            pred_tgt_son2 = self.cls_fc_son2(data_tgt_son2)
            data_tgt_son3 = self.sonnet3(data_tgt)
            pred_tgt_son3 = self.cls_fc_son3(data_tgt_son3)

            if mark == 1:
                data_src = self.mlp1(data_src)
                data_src = self.sonnet1(data_src)  # source1 features
                # mmd_loss += mmd.mmd(data_src, data_tgt_son1)
                mmd_loss = mmd(data_src, data_tgt_son1)  # mmd1
                l1_loss = torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1)  # 1-2  1-3
                                               - torch.nn.functional.softmax(data_tgt_son2, dim=1)))
                l1_loss += torch.mean(torch.abs(torch.nn.functional.softmax(data_tgt_son1, dim=1)
                                                - torch.nn.functional.softmax(data_tgt_son3, dim=1)))
                pred_src = self.cls_fc_son1(data_src)
                # t=F.log_softmax(pred_src, dim=1)###
                # print(t)####
                cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src)
                # cls_loss = nn.CrossEntropyLoss(F.log_softmax(pred_src, dim=1), label_src)
                return cls_loss, mmd_loss, l1_loss / 2, data_src, data_tgt_son1
