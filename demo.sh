#!/bin/bash

#--------------train--------------
# to visualize on web, you have to start visdom server. Result is on localhost:8900
# 1. create screen: 
#    screen -S visdom.8900
# 2. start visdom server:
#    python -m visdom.server -p 8900
# 3. leave screen: 
#    ctrl + a + d
#
# 4. uncomment to train
#
# python multi_label_classifier.py --dir "./test/celeba/" --mode "Train" --model "LightenB" --name "test" --batch_size 64 --gpu_ids 0 --input_channel 3 --load_size 144 --input_size 128 --mean [0,0,0] --std [1,1,1] --ratio "[0.94, 0.03, 0.03]" --shuffle --load_thread 8 --sum_epoch 5 --lr_decay_in_epoch 1 --display_port 8900 --validate_ratio 0.1 --top_k "(1,)" --score_thres 0.1 --display_train_freq 20 --display_validate_freq 20 --save_epoch_freq 1  --display_image_ratio 0.2
#
# 5. open localhost:8900 on your browser and you will see loss and accuracy curves and training images samples later on.
#
#--------------test--------------
# 6. uncomment to test
#
# python multi_label_classifier.py --dir "./test/celeba/" --mode "Test" --model "LightenB" --name "test" --batch_size 1 --gpu_ids 4 --input_channel 3 --load_size 144 --input_size 128 --mean [0,0,0] --std [1,1,1] --shuffle --load_thread 1 --top_k "(1,2)" --score_thres "0.1" --checkpoint_name "epoch_2_snapshot.pth" 
#
#-------------deploy-------------
# 7. If you have to classify a dataset neither in train set nor in test set, you can use deploy.py to do so. You have to specify label.txt and data_dir where data to be classified exists
#
# python deploy.py --dir "./test/celeba/" --data_dir "test/celeba/deploy/" --label_file "test/celeba/label.txt"  --mode "Test" --model "LightenB" --name "test" --input_channel 3 --gpu_ids 2 --load_size 144 --input_size 128 --checkpoint_name "epoch_2_snapshot.pth"  
