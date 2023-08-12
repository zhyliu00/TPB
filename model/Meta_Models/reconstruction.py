import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import time
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch.nn.utils import weight_norm
import math
from meta_patch import TimeBlock
        
class ReconstrucAdjNet(nn.Module):
    def __init__(self,indim):
        super(ReconstrucAdjNet,self).__init__()
        self.model = nn.Linear(indim, indim)
        self.Qnet = nn.Linear(indim, indim)
        self.Knet = nn.Linear(indim, indim)
        self.temperature = indim ** 0.25
        # self.temperature = 1
    # def forward(self, emb):
    #     '''
    #     simply inner product
    #     '''
    #     B, N, D = emb.shape
        
    #     # q,k : [B, N, D]
    #     q,k = emb, emb
    #     # A : [B, N, N]
    #     A = torch.einsum('bik,bkj->bij',q,k.permute(0,2,1))
        
        
    #     # A = F.softmax(res / self.temperature, dim=2)  
    #     A = F.softmax(A / self.temperature, dim=2)  
    #     # print(torch.max(A[0],dim=1)[0])
    #     return A
    
    
    # def forward(self, emb):
    #     '''
    #     Q-K inner product graph reconstruction with k nearest
    #     emb : the input embedding to reconstruct the A.
    #           Size [B, N, D]
    #     '''        
        
    #     B, N, D = emb.shape
    #     # q,k : [B, N, D]
    #     q = self.Q(emb)
    #     k = self.K(emb)
    #     # A : [B, N, N]
    #     A = torch.einsum('bik,bkj->bij',q,k.permute(0,2,1))
        
    #     v, idx = torch.topk(A,k=self.k,dim=2)
    #     res = torch.zeros(A.shape).to(A.device)
    #     res = res.scatter(2, idx, v)
    #     # print(res)
    #     # A = F.softmax(res / self.temperature, dim=2)  
    #     A = F.softmax(res, dim=2)  
    #     # print(A)
        
    #     return A
    
    def forward(self, emb):
        '''
        Q-K inner product graph reconstruction
        emb : the input embedding to reconstruct the A.
              Size [B, N, D]
        '''        
        B, N, D = emb.shape
        # q,k : [B, N, D]
        q = self.Qnet(emb)
        k = self.Knet(emb)
        # A : [B, N, N]
        A = torch.einsum('bik,bkj->bij',q,k.permute(0,2,1))
        A = F.softmax(A / self.temperature, dim=2)  
        return A
        
    # def forward(self, emb):
    #     '''
    #     inner product graph reconstruction
    #     emb : the input embedding to reconstruct the A.
    #           Size [B, N, D]
    #     '''
    #     B, N, D = emb.shape
    #     # h : [B, N, D]
    #     h = self.model(emb)
    #     # A : [B, N, N]
    #     A = torch.einsum('bik,bkj->bij',h,h.permute(0,2,1))
    #     A = F.softmax(A / self.temperature, dim=2)  
    #     return A
        
    # def forward(self,emb):
    #     '''
    #     GAT-Like graph reconstruction
    #     emb : the input embedding to reconstruct the A.
    #           Size [B, N, D]
    #     '''
    #     B, N, D = emb.shape
    #     # h : [B, N, D]
    #     h = self.model(emb)
    #     hr = h.repeat(1,1,N).view(B, N * N, D)
    #     hr2 = h.repeat(1,N,1)
    #     # h : [B, N, N, 2*D]
    #     h = torch.cat([hr, hr2],dim=2).view(B,N, N, 2 * D)
    #     # A : [B, N, N]
    #     A = self.a_net(h).squeeze(-1)
    #     A = self.activation(A)
    #     A = F.softmax(A, dim=2)    
        
    #     return A
    


class BatchA_STGCNBlock(nn.Module):
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
        super(BatchA_STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
                                                     spatial_channels))
        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        # self.batch_norm = nn.BatchNorm2d(num_nodes)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat : [B, N, N] adj block
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # print("[Block] X shape is", X.shape)
        t = self.temporal1(X)
        # print("[Block] t1 shape is", t.shape)
        # print("t shape is {}, A_hat shape is {}".format(t.shape, A_hat.shape))
        
        # A_hat : [B, N, N]
        # t : [B, N, L, D]
        lfs = torch.einsum("kij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # print("lfs shape is {}".format(lfs.shape))
        t2 = F.tanh(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        # print("[Block] t2 shape is", t2.shape)
        t3 = self.temporal2(t2)
        # print("[Block] t3 shape is", t3.shape)
        # return self.batch_norm(t3)
        return t3


    
class BatchA_STGCN(nn.Module):
    def __init__(self, model_args, task_args,args):
        super(BatchA_STGCN, self).__init__()
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
        self.block1 = BatchA_STGCNBlock(in_channels=self.message_dim, out_channels=self.hidden_dim, spatial_channels=self.hidden_dim)
        self.block2 = BatchA_STGCNBlock(in_channels=self.hidden_dim, out_channels=self.hidden_dim, spatial_channels=self.hidden_dim)
        # self.last_temporal = TimeBlock(in_channels=self.hidden_dim, out_channels=self.hidden_dim)
        # print("[{}, {}, {}]".format(self.his_num ,self.hidden_dim,self.pred_num))
        self.fully = nn.Linear((self.his_num - 4 * 2) * self.hidden_dim,self.pred_num)
        # self.fully = nn.Linear((self.his_num - 2 * 5) * self.hidden_dim,
        #                        self.pred_num)
    
    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix. [B, N, N]
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

     
class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class BatchA_nconv(nn.Module):
    def __init__(self):
        super(BatchA_nconv,self).__init__()

    def forward(self,x, A):
        # here A : [B, N, N]
        # x : [B, dilation_channels, N, L2]
        
        # this step : [B, D, N, L] * [B, N, N] -> [B, D, N, L]
        x = torch.einsum('ncvl,nwv->ncwl',(x,A))
        return x.contiguous()    

class BatchA_gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(BatchA_gcn,self).__init__()
        self.nconv = BatchA_nconv()
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


class BatchA_patch_gwnet(nn.Module):
    def __init__(self, dropout=0.3, gcn_bool=True, in_dim=1,out_dim=12,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2,supports_len=2):
        super(BatchA_patch_gwnet, self).__init__()
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
                    self.gconv.append(BatchA_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field

        # self.final_conv = nn.Linear(out_dim * 116,
        #                             out_dim,)
        

        
        
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
                    
        return x, supports
