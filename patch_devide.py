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
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('./model')
sys.path.append('./model/TSFormer')
sys.path.append('./model/Meta_Models')
from meta_patch import *
from TSmodel import *
from utils import *
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='metr-la', type=str)
parser.add_argument('--data_list', default='chengdu_shenzhen_metr',type=str)
parser.add_argument('--gpu', default=1, type = int)

args = parser.parse_args()

set_seed(7)
torch.set_default_dtype(torch.float32)
args.gpu=0

if __name__ == "__main__":
    if torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
    args.device = torch.device('cpu')
    print("INFO: {}".format(args.device))

    with open(args.config_filename) as f:
        config = yaml.load(f)

    torch.manual_seed(7)

    data_args, task_args, model_args = config['data'], config['task'], config['model']
    
    model = TSFormer(model_args['mae']).to(args.device)
    model.mode = 'Test'
    data_list = get_data_list(args.data_list)
    model_path = Path('./save/pretrain_model/{}/best_model.pt'.format(args.data_list))
    model.load_state_dict(torch.load(model_path))
    
    
    source_dataset = traffic_dataset(data_args, task_args['mae'], data_list, "cluster", test_data=args.test_dataset)
    
    # due to the memory consumption, divide the data.
    data_divide = 288
    patch_pool = None
    emb_pool = None
    unnorm_patch_pool = None
    source_dataset.data_list = np.flip(source_dataset.data_list)
    for dataset in source_dataset.data_list:
        print(dataset)
        # [N, 2, L]
        # x = torch.tensor(source_dataset.x_list[dataset],dtype=torch.float32).to(args.device)
        x = source_dataset.x_list[dataset]
        means, stds = source_dataset.means_list[dataset], source_dataset.stds_list[dataset]
        P = 12
        if(dataset == args.test_dataset):
            target_days = 3
            x = x[:,:, :288 * target_days]
        x, y = generate_dataset(x,288,0,means,stds,12)
        B, N, C, L = x.shape
        print("x shape : {}".format(x.shape))
        
        # select some of the batch to form the pattern
        length = B
        select_batch = 48
        rand_perm =torch.randperm(length)
        select_rand_perm = rand_perm[:select_batch]
        print(select_rand_perm)
        for idx in select_rand_perm:
            # [B, N, L, 2]
            # remember that the input of TSFormer is [B, N, 2, L]
            part_x = x[idx:idx+1].float().to(args.device).permute(0,1,3,2) 
            

            
            print(part_x.shape, part_x.dtype)
            H = model(part_x)
            B, N, C, L = part_x.shape
            unnorm_part_x = copy.deepcopy(part_x[:,:,0,:].reshape(B*N*int(L//P),P))
            part_x = part_x[:,:,0,:].reshape(B*N*int(L//P),P)
            part_x = part_x * stds[0] + means[0]
            H = H.reshape(B*N* int(L//P), -1)

            print('devided x shape : {}'.format(part_x.shape))
            print('devided normed_x shape : {}'.format(unnorm_part_x.shape))
            print('corresbonding H shape : {}'.format(H.shape))
            
            if(patch_pool == None):
                patch_pool = part_x.cpu()
                emb_pool = H.cpu()
                unnorm_patch_pool = unnorm_part_x.cpu()
            else:
                patch_pool = torch.cat([patch_pool.cpu(),part_x.cpu()],dim = 0)
                unnorm_patch_pool = torch.cat([unnorm_patch_pool.cpu(),unnorm_part_x.cpu()],dim = 0)
                emb_pool = torch.cat([emb_pool.cpu(), H.cpu()], dim = 0)
                    
            print("patch_pool shape : {}, emb_pool shape : {}".format(patch_pool.shape, emb_pool.shape))

            # [N, 2, L/B, B]
    print("patch_pool shape : {}, emb_pool shape : {}, unnorm_patch_pool shape {}".format(patch_pool.shape, emb_pool.shape, unnorm_patch_pool.shape))
    if (not os.path.exists('./pattern/{}/'.format(args.data_list))):
        os.makedirs('./pattern/{}/'.format(args.data_list))
    torch.save(patch_pool,'./pattern/{}/patch.pt'.format(args.data_list))
    torch.save(unnorm_patch_pool,'./pattern/{}/unorm_patch.pt'.format(args.data_list))
    torch.save(emb_pool,'./pattern/{}/emb.pt'.format(args.data_list))