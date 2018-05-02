#!/bin/bash

#--------------train--------------
# to visualize on web, you have to start visdom server. Result is on localhost:8900
# 1. create screen: 
#    screen -S visdom.8900
# 2. start visdom server:
#    python -m visdom.server -p 8900
# 3. leave screen: 
#    ctrl + a + d
# 4. start demo.sh
#python multi_label_classifier.py --dir "./test/celeba/" --mode "Train" --name "test" --batch_size 64 --gpu_ids 0 --input_channel 3 --load_size 144 --input_size 128 --mean [0,0,0] --std [1,1,1] --ratio "[0.94, 0.03, 0.03]" --shuffle --load_thread 8 --sum_epoch 20 --lr_decay_in_epoch 4 --display_port 8900 --validate_ratio 0.1 --top_k "(1,)" --score_thres 0.1 --display_train_freq 20 --display_validate_freq 20 --save_epoch_freq 1  --display_image_ratio 0.2
# 5. open localhost:8900 on your browser and you will see loss and accuracy curves and training images samples later on.


#--------------test--------------
python multi_label_classifier.py --dir "./test/celeba/" --mode "Test" --name "test" --batch_size 1 --gpu_ids 4 --input_channel 3 --load_size 144 --input_size 128 --mean [0,0,0] --std [1,1,1] --shuffle --load_thread 1 --top_k "(1,2)" --score_thres 0.1 --checkpoint_name "epoch_1_snapshot.pth" 
