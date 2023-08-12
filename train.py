import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import DataLoader
from datasets import traffic_dataset
from utils import *
import argparse
import yaml
import time
from tqdm import tqdm
import sys
sys.path.append('./model')
sys.path.append('./model/TSFormer')
sys.path.append('./model/Meta_Models')
from meta_patch import *
from reconstruction import *
from TSmodel import *
# from maml_model import *
from rep_model_final import *
from pathlib import Path
import random


parser = argparse.ArgumentParser(description='TPB')
parser.add_argument('--config_filename', default='./configs/config_pems.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='pems-bay', type=str)
parser.add_argument('--train_epochs', default=200, type=int)
parser.add_argument('--finetune_epochs', default=120,type=int)
parser.add_argument('--lr',default=1e-3,type=float)
parser.add_argument('--update_step', default=5,type=int)

parser.add_argument('--seed',default=7,type=int)
parser.add_argument('--data_list', default='chengdu_shenzhen_metr',type=str)
parser.add_argument('--target_days', default=3,type=int)
parser.add_argument('--patch_encoder', default='pattern', type=str)
parser.add_argument('--gpu', default=0, type = int)
parser.add_argument('--sim', default='cosine', type = str)
parser.add_argument('--K', default=10, type = int)
parser.add_argument('--STmodel',default='GWN',type=str)
parser.add_argument('--his_num',default=288,type=int)
args = parser.parse_args()

args.new=1
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float32)

# since historical 1 day data is used to generate metaknowledge
print(time.strftime('%Y-%m-%d %H:%M:%S'), "Forecasting target_days = {}".format(args.target_days - 1))

if __name__ == '__main__':
    if torch.cuda.is_available():
        args.device = torch.device('cuda:0')
        print("INFO: GPU : {}".format(args.gpu))
    else:
        args.device = torch.device('cpu')
        print("INFO: CPU")

    with open(args.config_filename) as f:
        config = yaml.load(f)


    print(config)
    args.data_list = config['model']['STnet']['data_list']
    args.batch_size = config['task']['maml']['batch_size']
    args.test_dataset = config['task']['maml']['test_dataset']
    args.K = config['model']['STnet']['K']
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    data_list = get_data_list(args.data_list)
    print("INFO: train on {}. test on {}.".format(data_list, args.test_dataset))
    PatchFSL_cfg = {
        'data_list' : args.data_list,
        'sim': args.sim,
        'K' : args.K,
        'patch_encoder': args.patch_encoder,
        'base_dir': Path(sys.path[0]),
        'device':args.device
    }

    ## dataset
    source_dataset = traffic_dataset(data_args, task_args['maml'], data_list, "source_train", test_data=args.test_dataset)
    ## check source dataset

    for data in source_dataset.data_list:
        print("source dataset has {}. X : {}, y : {}".format(data,source_dataset.x_list[data].shape,source_dataset.y_list[data].shape))
    
    finetune_dataset = traffic_dataset(data_args, task_args['maml'], data_list, 'target_maml', test_data=args.test_dataset)
    test_dataset = traffic_dataset(data_args, task_args['maml'], data_list, 'test', test_data=args.test_dataset)
    print(data_args, task_args, model_args, PatchFSL_cfg, args.STmodel)
    rep_model = STRep(data_args, task_args, model_args, PatchFSL_cfg, args.STmodel)
    best_loss = 9999999999999.0
    best_model = None
    ## train on big dataset
    rep_tasknum = task_args['maml']['task_num']

    for i in range(task_args['maml']['train_epochs']):
        length = source_dataset.__len__()
        # length=40
        print('----------------------')
        time_1 = time.time()
        
        data_spt = []
        matrix_spt = []
        data_qry = []
        matrix_qry = []

        idx = 0
        for jj in range(rep_tasknum):
            data_i, A = source_dataset[idx]
            data_spt.append(data_i)
            matrix_spt.append(A)
            idx+=1

            data_i, A = source_dataset[idx]
            data_qry.append(data_i)
            matrix_qry.append(A)
            idx+=1
        
        model_loss ,mse_loss, rmse_loss, mae_loss, mape_loss = rep_model.meta_train_revise(data_spt, matrix_spt, data_qry, matrix_qry)

        print('Epochs {}/{}'.format(i,task_args['maml']['train_epochs']))
        print('in meta-training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, reconstruction Loss : {:.5f}.'.format(mse_loss, rmse_loss, mae_loss,mape_loss,model_loss))
        print("This epoch cost {:.3}s.".format(time.time() - time_1))
    rep_model.finetuning(finetune_dataset, test_dataset, task_args['maml']['finetune_epochs'])
