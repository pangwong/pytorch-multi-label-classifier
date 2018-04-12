#!/bin/bash

# to visualize on web, you have to start visdom server
python -m visdom.server -p 8900

python train.py --dir "./test/mnist/" --mode "Train" --name "test1" --batch_size 16 --gpu_ids 7 --input_channel 1 --load_size 144 --input_size 128 --mean [0] --std [1] --display_validate_freq 50 --display_train_freq 10 --display_data_freq 10 --ratio "[0.95, 0.005,0.045]" --score_thres 0.1 --top_k "(1,2)" --display_port 8900 --validate_ratio 0.1 --sum_epoch 50 --save_epoch_freq 1 
