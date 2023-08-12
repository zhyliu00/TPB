import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from datasets import traffic_dataset
from utils import *
import argparse
import yaml
import time
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('./model')
sys.path.append('./model/TSFormer')
sys.path.append('./model/Meta_Models')
from meta_patch import *
from TSmodel import *
import copy
# from pykeops.torch import LazyTensor
# import faiss
from kmeans_pytorch import kmeans

parser = argparse.ArgumentParser()
parser.add_argument('--data_list', default='chengdu_shenzhen_metr',type=str)
parser.add_argument('--gpu', default=0, type = int)
parser.add_argument('--sim', default='cosine',type = str)
parser.add_argument('--K', default=1000, type = int)
args = parser.parse_args()
args.gpu=0

use_cuda = False

# def KMeans(x, K=10, Niter=10, verbose=True):
#     """Implements Lloyd's algorithm for the Euclidean metric."""

#     start = time.time()
#     N, D = x.shape  # Number of samples, dimension of the ambient space

#     c = x[:K, :].clone()  # Simplistic initialization for the centroids

#     x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
#     c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids
#     # x_i = x.view(N, 1, D)
#     # c_j = c.view(1, K, D)

#     # K-means loop:
#     # - x  is the (N, D) point cloud,
#     # - cl is the (N,) vector of class labels
#     # - c  is the (K, D) cloud of cluster centroids
#     for i in range(Niter):
#         print(i)
#         # E step: assign points to the closest cluster -------------------------
#         D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
#         cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

#         # M step: update the centroids to the normalized cluster average: ------
#         # Compute the sum of points per cluster:
#         c.zero_()
#         c.scatter_add_(0, cl[:, None].repeat(1, D), x)

#         # Divide by the number of points per cluster:
#         Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
#         c /= Ncl  # in-place division to compute the average
        
#         del Ncl,D_ij
#     if verbose:  # Fancy display -----------------------------------------------
#         if use_cuda:
#             torch.cuda.synchronize()
#         end = time.time()
#         print(
#             f"K-means for the Euclidean metric with {N:,} points in dimension {D:,}, K = {K:,}:"
#         )
#         print(
#             "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
#                 Niter, end - start, Niter, (end - start) / Niter
#             )
#         )

#     return cl, c


# def KMeans_cosine(x, K=10, Niter=10, verbose=True):
#     """Implements Lloyd's algorithm for the Cosine similarity metric."""

#     start = time.time()
#     N, D = x.shape  # Number of samples, dimension of the ambient space

#     c = x[:K, :].clone()  # Simplistic initialization for the centroids
#     # Normalize the centroids for the cosine similarity:
#     c = torch.nn.functional.normalize(c, dim=1, p=2)

#     x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
#     c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

#     # K-means loop:
#     # - x  is the (N, D) point cloud,
#     # - cl is the (N,) vector of class labels
#     # - c  is the (K, D) cloud of cluster centroids
#     for i in range(Niter):

#         # E step: assign points to the closest cluster -------------------------
#         S_ij = x_i | c_j  # (N, K) symbolic Gram matrix of dot products
#         cl = S_ij.argmax(dim=1).long().view(-1)  # Points -> Nearest cluster

#         # M step: update the centroids to the normalized cluster average: ------
#         # Compute the sum of points per cluster:
#         c.zero_()
#         c.scatter_add_(0, cl[:, None].repeat(1, D), x)

#         # Normalize the centroids, in place:
#         c[:] = torch.nn.functional.normalize(c, dim=1, p=2)

#     if verbose:  # Fancy display -----------------------------------------------
#         if use_cuda:
#             torch.cuda.synchronize()
#         end = time.time()
#         print(
#             f"K-means for the cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:"
#         )
#         print(
#             "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
#                 Niter, end - start, Niter, (end - start) / Niter
#             )
#         )

#     return cl, c



if __name__ == "__main__":
    if torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.gpu))
        use_cuda = True
    else:
        args.device = torch.device('cpu')
    # args.device = torch.device('cpu')
    print("INFO: {}".format(args.device))
    
    patch_pool = torch.load('./pattern/{}/patch.pt'.format(args.data_list)).to(args.device)
    unnorm_patch_pool = torch.load('./pattern/{}/unorm_patch.pt'.format(args.data_list)).to(args.device)
    emb_pool = torch.load('./pattern/{}/emb.pt'.format(args.data_list)).to(args.device)
    
    N,D = emb_pool.shape
    sample_ratio = 0.1
    indices = torch.randperm(N)[:int(N * sample_ratio)]
    emb_pool = emb_pool[indices]
    print("Use {}. After sampling {}, N = {}".format(args.sim,sample_ratio, emb_pool.shape[0]))
    # if(args.sim == 'euclidean'):
    #     c, cl = KMeans(emb_pool,args.K,50)
    # elif(args.sim == 'cosine'):
    #     c, cl = KMeans_cosine(emb_pool,args.K,50)
    # print(c.shape, cl.shape, args.K)
    
    
    
    # ncentroids = args.K
    # niter = 150
    # verbose = True
    # d = emb_pool.shape[1]
    # kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
    # res = kmeans.train(emb_pool)
    # cl = res.centroids
    # c, cl = KMeans(
    #     X=emb_pool, num_clusters=args.K, distance=args.sim, device=torch.device('cuda:0')
    # )
    
    
    c, cl = kmeans(
        X=emb_pool, num_clusters=args.K, distance=args.sim, device=torch.device('cuda:0')
    )
    torch.save(c,'./pattern/{}/{}_{}_c.pt'.format(args.data_list,args.sim,args.K))
    torch.save(cl,'./pattern/{}/{}_{}_cl.pt'.format(args.data_list,args.sim,args.K))
    
    
    
