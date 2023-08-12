OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  --config_file ./configs/config_pems.yaml > train_pems.out 2>&1 &
wait
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  --config_file ./configs/config_metr.yaml > train_metr.out 2>&1 &
wait
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  --config_file ./configs/config_chengdu.yaml > train_chengdu.out 2>&1 &
wait
OMP_NUM_THREADS=64 CUDA_VISIBLE_DEVICES=0 nohup python -u train.py  --config_file ./configs/config_shenzhen.yaml > train_shenzhen.out 2>&1 &
wait