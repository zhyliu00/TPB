from cProfile import label
from macpath import split
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

parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', default='config.yaml', type=str,
                        help='Configuration filename for restoring the model.')
parser.add_argument('--test_dataset', default='pems-bay', type=str)
parser.add_argument('--data_list', default='chengdu',type=str)
parser.add_argument('--gpu', default=0, type = int)
args = parser.parse_args()
args.gpu=0
seed=7
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_default_dtype(torch.float32)
# print(time.strftime('%Y-%m-%d %H:%M:%S'), "meta_dim = ", args.meta_dim,"target_days = ", args.target_days)



def train_batch(start,end,model,source_dataset,loss_fn,opt):
    total_loss = []
    total_mae = []
    total_mse = []
    total_rmse = []
    total_mape = []
    model.train()
    for idx in range(start,end):
        data_i, A_wave = source_dataset[idx]
        # [B, N, L, 2]
        x, means, stds = data_i.x, data_i.means, data_i.stds
        
        # remember that the input of TSFormer is [B, N, 2, L]
        x = x.permute(0,1,3,2).to(args.device)
        # 
        out_masked_tokens, label_masked_tokens, plot_args = model(x)
        
        # only the masked patch is loss target 
        loss = loss_fn(out_masked_tokens, label_masked_tokens)
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # unmask
        unnorm_out, unnorm_label = unnorm(out_masked_tokens, means, stds), unnorm(label_masked_tokens,means,stds)
        # print(unnorm_out.shape, unnorm_label.shape)
        # unnorm_out, unnorm_label = unnorm_out.cpu().detach().numpy(), unnorm_label.cpu().detach().numpy()
        MSE,RMSE,MAE,MAPE = calc_metric(unnorm_out, unnorm_label)
        
        total_mse.append(MSE.cpu().detach().numpy())
        total_rmse.append(RMSE.cpu().detach().numpy())
        total_mae.append(MAE.cpu().detach().numpy())
        total_mape.append(MAPE.cpu().detach().numpy())
        total_loss.append(loss.item())
    return total_mse,total_rmse, total_mae, total_mape, total_loss
        # print(out_masked_tokens.shape, label_masked_tokens.shape, list(plot_args.keys()))

def test_batch(start,end,model,source_dataset,loss_fn,opt):
    total_loss = []
    total_mae = []
    total_mse = []
    total_rmse = []
    total_mape = []
    with torch.no_grad():
        model.eval()
        for idx in range(start,end):
            data_i, A_wave = source_dataset[idx]
            # [B, N, L, 2]
            x, means, stds = data_i.x, data_i.means, data_i.stds
            
            # remember that the input of TSFormer is [B, N, 2, L]
            x = x.permute(0,1,3,2).to(args.device)
            # 
            out_masked_tokens, label_masked_tokens, plot_args = model(x)
            # unmask
            unnorm_out, unnorm_label = unnorm(out_masked_tokens, means, stds), unnorm(label_masked_tokens,means,stds)
            MSE,RMSE,MAE,MAPE = calc_metric(unnorm_out, unnorm_label)

            total_mse.append(MSE.cpu().detach().numpy())
            total_rmse.append(RMSE.cpu().detach().numpy())
            total_mae.append(MAE.cpu().detach().numpy())
            total_mape.append(MAPE.cpu().detach().numpy())
        return total_mse,total_rmse, total_mae, total_mape, total_loss

if __name__ == "__main__":
    if torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')
    print("INFO: {}".format(args.device))

    with open(args.config_filename) as f:
        config = yaml.load(f)
        
    data_args, task_args, model_args = config['data'], config['task'], config['model']
    train_ratio = 0.7
    val_ratio = 0.1
    test_ratio = 0.2
    
    model_path = Path('./save/pretrain_model/{}/'.format(args.data_list))
    if(not os.path.exists(model_path)):
        os.makedirs(model_path)
    data_list = get_data_list(args.data_list)
    
    
    source_dataset = traffic_dataset(data_args, task_args['mae'], data_list, "pretrain", test_data=args.test_dataset)

    model = TSFormer(model_args['mae']).to(args.device)
    model.mode = 'Pretrain'
    opt = optim.Adam(model.parameters(),lr = task_args['mae']['lr'])
    loss_fn = nn.MSELoss(reduction = 'mean')
    batch_size = task_args['mae']['batch_size']
    print('pretrain model has {} parameters'.format(count_parameters(model)))
    
    best_loss = 9999999999999.0
    best_model = None
    for i in range(task_args['mae']['train_epochs']):
        length = source_dataset.__len__()
        # length=40
        train_length = int(train_ratio * length)
        val_length = int(val_ratio * length)
        
        print('----------------------')
        time_1 = time.time()
        total_mse,total_rmse, total_mae, total_mape, total_loss = train_batch(0,train_length, model, source_dataset,loss_fn,opt)
        print('Epochs {}/{}'.format(i,task_args['mae']['train_epochs']))
        print('in training   Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}, normed MSE : {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape),np.mean(total_loss)))
        
        total_mse,total_rmse, total_mae, total_mape, total_loss = test_batch(train_length,train_length + val_length, model, source_dataset,loss_fn,opt)
        print('in validation Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}.'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)))
        
        mae_loss = np.mean(total_mae)
        if(mae_loss < best_loss):
            best_model = model
            best_loss = mae_loss
            torch.save(model.state_dict(), model_path / 'best_model.pt')
            print('Best model. Saved.')
        print('this epoch costs {:.5}s'.format(time.time()-time_1))
        
    total_mse,total_rmse, total_mae, total_mape, total_loss = test_batch(train_length + val_length,length,model, source_dataset,loss_fn,opt)
    print('test  Unnormed MSE : {:.5f}, RMSE : {:.5f}, MAE : {:.5f}, MAPE: {:.5f}'.format(np.mean(total_mse), np.mean(total_rmse), np.mean(total_mae),np.mean(total_mape)))