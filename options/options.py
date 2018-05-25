import os
import torch
import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        self.parser.add_argument('--dir', required=True, default='./', help='path to the data directory containing data.txt and label.txt')
        self.parser.add_argument('--name', required=True, default='test', help='subdirectory name for training or testing, snapshot, splited dataset and test results exist here')
        self.parser.add_argument('--mode', required=True, default='Train', help='run mode of training or testing. [Train | Test | train | test]')
        self.parser.add_argument('--model', required=True, default='alexnet', help='model type. [Alexnet | LightenB | VGG16 | Resnet18 | ...]')
        self.parser.add_argument('--load_size', type=int, default=144, help='scale image to the size prepared for croping')
        self.parser.add_argument('--input_size', type=int, default=128, help='then crop image to the size as network input')
        self.parser.add_argument('--ratio', type=str, default='[0.95, 0.025, 0.025]', help='ratio of whole dataset for Train, Validate, Test resperctively')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size of network input. Note that batch_size should only set to 1 in Test mode')
        self.parser.add_argument('--shuffle', action='store_true', help='default false. If true, data will be shuffled when split dataset and in batch')
        self.parser.add_argument('--flip', action='store_true', help='if true, flip image randomly before input into network')
        self.parser.add_argument('--region', action='store_false', help='if true, crop image by input box')
        self.parser.add_argument('--load_thread', type=int, default=2, help='how many subprocesses to use for data loading')
        self.parser.add_argument('--crop', type=str, default='CenterCrop', help='crop type, candidates are [NoCrop | RandomCrop | CenterCrop | FiveCrop | TenCrop]')
        self.parser.add_argument('--gray', action='store_true', help='defalut false. If true, image will be converted to gray_scale')
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--box_ratio', type=float, default=-1, help='modify box ratio of width and height to specified ratio')
        self.parser.add_argument('--box_scale', type=float, default=1.0, help='scale box to specified ratio. Default 1.0 means no change')
        self.parser.add_argument('--input_channel', type=int, default=3, help='set input image channel, 1 for gray and 3 for color')
        self.parser.add_argument('--mean', type=str, default='(0,0,0)', help='sequence of means for each channel used for normization')
        self.parser.add_argument('--std', type=str, default='(1,1,1)', help='sequence standard deviations for each channel used for normization')
        self.parser.add_argument('--padding', action='store_true', help='default false. If true, image will be padded if scaled box is out of image boundary')
        self.parser.add_argument('--checkpoint_name', type=str, default='', help='path to pretrained model or model to deploy')
        self.parser.add_argument('--pretrain', action='store_true', help='default false. If true, load pretrained model to initizaize model state_dict')
        ## for train
        self.parser.add_argument('--validate_ratio', type=float, default=1, help='ratio of validate set when validate model')
        self.parser.add_argument('--sum_epoch', type=int, default=200, help='sum epoches for training')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='save snapshot every $save_epoch_freq epoches training')
        self.parser.add_argument('--save_batch_iter_freq', type=int, default=100, help='save snapshot every $save_batch_iter_freq training') 
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay.')
        self.parser.add_argument('--lr_mult_w', type=float, default=20, help='learning rate of W of last layer parameter will be lr*lr_mult_w')
        self.parser.add_argument('--lr_mult_b', type=float, default=20, help='learning rate of b of last layer parameter will be lr*lr_mult_b')
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_in_epoch', type=int, default=50, help='multiply by a gamma every lr_decay_in_epoch iterations')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of SGD')
        self.parser.add_argument('--loss_weight', type=str, default='', help='list. Loss weight for cross entropy loss.For example set $loss_weight to [1, 0.8, 0.8] for a 3 labels classification')

        ## for test
        self.parser.add_argument('--top_k', type=str, default='(1,)', help='tuple. We only take top k classification results into accuracy consideration')
        self.parser.add_argument('--score_thres', type=str, default='0.0', help='float or list. We only take classification results whose score is bigger than score_thres into recall consideration')
        # these tow param below used only in deploy.py
        self.parser.add_argument('--label_file', type=str, default="", help='label file only for deploy a checkpoint model')
        self.parser.add_argument('--classify_dir', type=str, default="", help='directory where data.txt to be classified exists')
        
        ## for visualization
        self.parser.add_argument('--display_winsize', type=int, default=128, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display. Less than 1 will display nothing')
        self.parser.add_argument('--display_port', type=int, default=8097, help='port of visdom server for web display. Result will show on `localhost:$display_port`')
        self.parser.add_argument('--image_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--html', action='store_false', help='defalt true. Do not save intermediate training results to [opt.dir]/[opt.name]/web/')
        self.parser.add_argument('--update_html_freq', type=int, default=10, help='frequency of saving training results to html')
        self.parser.add_argument('--display_train_freq', type=int, default=10, help='print train loss and accuracy every $train_freq batches iteration')
        self.parser.add_argument('--display_validate_freq', type=int, default=10, help='test validate dateset every $validate_freq batches iteration')
        self.parser.add_argument('--display_data_freq', type=int, default=10, help='frequency of showing training data on web browser')
        self.parser.add_argument('--display_image_ratio', type=float, default=1.0, help='ratio of images in a batch showing on web browser')

    def parse(self):
        opt = self.parser.parse_args()
        
        # mode
        if opt.mode not in ["Train", "Test", "train", "test"]:
            raise Exception("cannot recognize flag `mode`")
        opt.mode = opt.mode.capitalize()
        if opt.mode == "Test":
            opt.batch_size = 1
            opt.shuffle = False

        # devices id
        gpu_ids = opt.gpu_ids.split(',')
        opt.devices = []
        for id in gpu_ids:
            if eval(id) >= 0:
                opt.devices.append(eval(id))
        # cuda
        opt.cuda = False
        if len(opt.devices) > 0 and torch.cuda.is_available():
            opt.cuda = True


        opt.top_k = eval(opt.top_k)
        opt.mean = eval(opt.mean)
        opt.std = eval(opt.std)
        opt.ratio = eval(opt.ratio)
        if opt.loss_weight == "":
            opt.loss_weight=None
        else:
            opt.loss_weight = torch.FloatTensor(eval(opt.loss_weight))

        return opt

if __name__ == "__main__":
    op = Options()
    op.parse()
