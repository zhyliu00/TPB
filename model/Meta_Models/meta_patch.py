import math
from tkinter import E
from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import time
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm
import copy
        
class FCNet(nn.Module):
    def __init__(self,indim,hiddim,outdim):
        super(FCNet,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(indim, hiddim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hiddim,hiddim),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hiddim,outdim)
        )
    def forward(self,x):
        return self.model(x)
    
class PatternEncoder(nn.Module):
    def __init__(self,pattern):
        super(PatternEncoder, self).__init__()
        # pattern [K, D]
        self.pattern = pattern
        self.K, self.D = pattern.shape
        self.temporature = np.sqrt(self.D)
        self.Qnet = nn.Linear(self.D, self.D)
        self.Knet = nn.Linear(self.D, self.D)
    def forward(self, X):
        """
        :param X: [B, N, L, D]. D is the size of embedding corresbonding to pattern.shape[1]
        """        
        # print(X.shape)
        # pattern : [K, D]
        pattern = self.Knet(self.pattern)
        # qry : [B, N, L, D]
        qry = self.Qnet(X)
        
        weight = torch.einsum('kd,bnld->bnlk',pattern,qry) / self.temporature
        # print(weight.shape)
        # [B, N, L, D]
        weight = F.softmax(weight,dim = 3)
        # print(weight.shape)
        # [B, N, L, D]
        w_X = torch.einsum('bnlk,kd->bnld',weight,self.pattern)
        # print(w_X.shape)
        return w_X
     
    
class PatternEncoder_patternkeyv1(nn.Module):
    def __init__(self,pattern):
        super(PatternEncoder_patternkeyv1, self).__init__()
        # pattern [K, D]
        self.pattern = pattern
        self.K, self.D = pattern.shape
        self.temporature = np.sqrt(self.D)
        self.Qnet = nn.Linear(self.D, self.D)
        self.Knet = nn.Linear(self.D, self.D)
        self.qry_mat = nn.Embedding(self.D, self.K)
    def forward(self, X):
        """
        :param X: [B, N, L, D]. D is the size of embedding corresbonding to pattern.shape[1]
        """        
        # print(X.shape)
        # pattern : [K, D]
        pattern = self.Knet(self.pattern)
        # qry : [B, N, L, D]
        qry = self.Qnet(X)
        
        
        weight = torch.einsum('kd,bnld->bnlk',pattern,qry) / self.temporature
        # print(weight.shape)
        # [B, N, L, D]
        weight = F.softmax(weight,dim = 3)
        # print(weight.shape)
        # [B, N, L, D]
        w_X = torch.einsum('bnlk,kd->bnld',weight,self.pattern)
        # print(w_X.shape)
        return w_X
     
class PatternEncoder_patternkeyv2(nn.Module):
    def __init__(self,pattern):
        super(PatternEncoder_patternkeyv2, self).__init__()
        # pattern [K, D]
        self.pattern = pattern
        self.K, self.D = pattern.shape
        self.temporature = np.sqrt(self.D)
        # self.Qnet = nn.Linear(self.D, self.D)
        self.Qnet = nn.Linear(12, self.D)
        self.Knet = nn.Linear(self.D, self.D)
        self.qry_mat = nn.Parameter(torch.randn(self.D,self.K))
    def forward(self, X):
        """
        :param X: [B, N, L, D]. D is the size of embedding corresbonding to pattern.shape[1]
        """        
        # print(X.shape)
        # pattern : [K, D]
    
        # qry : [B, N, L, D]
        qry = self.Qnet(X)
        
        
        
        # weight = torch.einsum('kd,bnld->bnlk',pattern,qry) / self.temporature
        weight = torch.einsum('dk,bnld->bnlk',self.qry_mat,qry)
        # print(weight.shape)
        # [B, N, L, D]
        weight = F.softmax(weight,dim = 3)
        # print(weight.shape)
        # [B, N, L, D]
        w_X = torch.einsum('bnlk,kd->bnld',weight,self.pattern)
        # print(w_X.shape)
        return w_X
    
    
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)
    
class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,wv->ncwl',(x,A))
        return x.contiguous()    

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out,dim=1)
        # print("after gconv out dim = {}, x dim = {}".format(h.shape, x2.shape))
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

        
        
class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        temp = self.conv1(X) + self.conv2(X)
        gate = torch.sigmoid(self.conv3(X))
        out = F.tanh(gate * temp)
        # temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        # out = F.relu(temp + self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    # def __init__(self, in_channels, spatial_channels, out_channels,
                #  num_nodes):
    def __init__(self, in_channels, spatial_channels, out_channels):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        # self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # print("[Block] X shape is", X.shape)
        t = self.temporal1(X)
        # print("[Block] t1 shape is", t.shape)
        # print("t shape is {}, A_hat shape is {}".format(t.shape, A_hat.shape))
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # print("lfs shape is {}".format(lfs.shape))
        t2 = F.tanh(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # print("[Block] t2 shape is", t2.shape)
        t3 = self.temporal2(t2)
        # print("[Block] t3 shape is", t3.shape)
        # return self.layer_norm(t3)

        return t3

class STGCN(nn.Module):
    def __init__(self, model_args, task_args,args):
        super(STGCN, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.message_dim = model_args['message_dim']
        # if args.patch_encoder == "raw":
        #     self.his_num = 12 
        # else:
        #     self.his_num = 128
        # if (args.new == 1):
        #     self.his_num = 12
        self.his_num = model_args['his_num']
        self.hidden_dim = model_args['hidden_dim']
        self.pred_num = task_args['pred_num']
        self.build()
    
    def build(self):
        self.block1 = STGCNBlock(in_channels=self.message_dim, out_channels=self.hidden_dim, spatial_channels=self.hidden_dim)
        self.block2 = STGCNBlock(in_channels=self.hidden_dim, out_channels=self.hidden_dim, spatial_channels=self.hidden_dim)
        # self.last_temporal = TimeBlock(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        # print("[{}, {}, {}]".format(self.his_num ,self.hidden_dim,self.pred_num))
        self.fully = nn.Linear((self.his_num - 4 * 2) * self.hidden_dim,self.pred_num)
        # self.fully = nn.Linear((self.his_num - 2 * 5) * self.hidden_dim,
        #                        self.pred_num)
    
    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        # print("x shape is", X.shape)
        out1 = self.block1(X, A_hat)
        # print("out1 shape is", out1.shape)
        out2 = self.block2(out1, A_hat)
        # print("out2 shape is", out2.shape)
        # out3 = self.last_temporal(out2)
        out3 = out2
        # print("out3 shape is", out3.shape)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        # print("out4 shape is", out4.shape)
        return out4, A_hat


class STGCN_baseline(nn.Module):
    def __init__(self):
        super(STGCN_baseline, self).__init__()
        self.message_dim = 1
        # if args.patch_encoder == "raw":
        #     self.his_num = 12 
        # else:
        #     self.his_num = 128
        # if (args.new == 1):
        #     self.his_num = 12
        self.his_num = 12
        self.hidden_dim = 256
        self.pred_num = 12
        self.build()
    
    def build(self):
        self.block1 = STGCNBlock(in_channels=self.message_dim, out_channels=self.hidden_dim, spatial_channels=self.hidden_dim)
        self.block2 = STGCNBlock(in_channels=self.hidden_dim, out_channels=self.hidden_dim, spatial_channels=self.hidden_dim)
        # self.last_temporal = TimeBlock(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        # print("[{}, {}, {}]".format(self.his_num ,self.hidden_dim,self.pred_num))
        self.fully = nn.Linear((self.his_num - 4 * 2) * self.hidden_dim,self.pred_num)
        # self.fully = nn.Linear((self.his_num - 2 * 5) * self.hidden_dim,
        #                        self.pred_num)
    
    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """
        # print("x shape is", X.shape)
        out1 = self.block1(X, A_hat)
        # print("out1 shape is", out1.shape)
        out2 = self.block2(out1, A_hat)
        # print("out2 shape is", out2.shape)
        # out3 = self.last_temporal(out2)
        out3 = out2
        # print("out3 shape is", out3.shape)
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1)))
        # print("out4 shape is", out4.shape)
        return out4

class patch_gwnet(nn.Module):
    def __init__(self, dropout=0.3, gcn_bool=True, in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2,supports_len=2):
        super(patch_gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        receptive_field = 1

        # All supports are double transition
        self.supports_len = supports_len

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                # print("receptive_filed : {}, addtional_scope : {}, new_dilation : {}".format(receptive_field, additional_scope, new_dilation))
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

        self.final_conv = nn.Linear(out_dim * 116,
                                    out_dim,)
        

        
        
    def forward(self, input, supports):
        if(not isinstance(supports, list)):
            supports = [supports]
        # input : [B, N, L, D]
        input = input.permute(0,3,1,2)
        # input : [B, D, N, L]
        in_len = input.size(3)
        if in_len<self.receptive_field:
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
        else:
            x = input
        # print("x shape is : {}".format(x.shape))
        
        x = self.start_conv(x)
        # x : [B, residual_channels, N, L]
        skip = 0       

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            # residual = x
            
            ## here maybe the author write wrong code
            
            residual = x.clone()
            
            
            # dilated convolution
            # print("Conv2d input shape is ", residual.shape)
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            # print("Conv1d input shape is ", residual.shape)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # x : [B, dilation_channels, N, L2]
            
            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            # s : [B, dilation_channels, N, L2]
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip
            
            # skip = 0 thus skip = s

            if self.gcn_bool and supports is not None:
                x = self.gconv[i](x,supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]


            x = self.bn[i](x)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        # print("x shape is : {}".format(x.shape))

        if(x.shape[-1]==1):
            x = (x.squeeze(-1)).permute(0,2,1)
        else:
            x = x.permute(0,2,1,3)
            x = torch.flatten(x, start_dim=2)
            x = self.final_conv(x)
                    
        return x, None
