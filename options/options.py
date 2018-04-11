import os
import torch
import argparse

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        ## add parse
        self.parser.add_argument('--dir', required=True, default='./', help="path to the data directory containing data.txt and label.txt")
        self.parser.add_argument('--name', required=True, default="test", help="subdirectory name for training or testing, snapshot or test results exist here")
        self.parser.add_argument('--mode', required=True, default='Train', help="run mode of training or testing. [Train | Test | train | test]")
        self.parser.add_argument('--load_size', type=int, default=144, help="scale image to the size prepared for croping")
        self.parser.add_argument('--input_size', type=int, default=128, help="then crop image to the size being network input")
        self.parser.add_argument('--ratio', type=str, default="[0.95, 0.025, 0.025]", help="ratio in whole dataset of Train, Validate, Test resperctively")
        self.parser.add_argument('--batch_size', type=int, default=1, help="batch size of network input. Note that batch_size should only set to 1 in Test mode")
        self.parser.add_argument('--shuffle', action='store_true', help="if true, data will be shuffled before input into network")
        self.parser.add_argument('--flip', action='store_true', help="if true, flip image randomly before input into network")
        self.parser.add_argument('--region', action='store_false', help="if true, crop image by input box")
        self.parser.add_argument('--load_thread', type=int, default=2, help="how many subprocesses to use for data loading")
        self.parser.add_argument('--crop', type=str, default="CenterCrop", help="crop type, candidates are [NoCrop | RandomCrop | CenterCrop | FiveCrop | TenCrop]")
        self.parser.add_argument('--gray', action='store_true', help="if true, image will be converted to gray_scale")
        self.parser.add_argument('--gpu_ids', type=str, default='-1', help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        self.parser.add_argument('--box_ratio', type=float, default=1.0, help="fix ratio of width and height to specified ratio")
        self.parser.add_argument('--box_scale', type=float, default=1.1, help="scale box to specified ratio")
        self.parser.add_argument('--input_channel', type=int, default=3, help="input image channel")
        self.parser.add_argument('--mean', type=str, default="(0,0,0)", help="sequence of means for each channel used for normization")
        self.parser.add_argument('--std', type=str, default="(1,1,1)", help="sequence standard deviations for each channel used for normization")
        self.parser.add_argument('--padding', action='store_true', help="if true, image will be padded if scaled box is out of image boundary")
        self.parser.add_argument('--checkpoint_name', type=str, default="", help="path to finetuning model or model to deploy")
        
        ## for train
        self.parser.add_argument('--fc_', type=float, default=1.0, help="")
        self.parser.add_argument('--validate_ratio', type=float, default=1, help='validate ratio of validate set')
        #self.parser.add_argument('--', type=float, default=1.0, help="fix ratio of width and height to specified ratio")
        self.parser.add_argument('--sum_epoch', type=int, default=200, help="sum epoches for training")
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help="save snapshot every $save_epoch_freq epoches training")
        self.parser.add_argument('--save_batch_iter_freq', type=int, default=100, help='save snapshot every $save_batch_iter_freq training') 
        
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='multiplicative factor of learning rate decay.')
        
        self.parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_in_epoch', type=int, default=50, help='multiply by a gamma every lr_decay_in_epoch iterations')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of SGD')
        self.parser.add_argument('--loss_weight', type=str, default='', help='loss weight for cross entropy loss.For example set $loss_weight to [1, 0.8, 0.8] for a 3 labels classification')

        ## for test
        self.parser.add_argument('--top_k', type=str, default="(1,)", help="we only take top k classification results into accuracy consideration")
        self.parser.add_argument('--score_thres', type=float, default=0.0, help="we only take classification results whose score is bigger than score_thres into recall consideration")
        
        ## for visualization
        self.parser.add_argument('--display_winsize', type=int, default=128, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--image_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--html', action='store_false', help='do not save intermediate training results to [opt.dir]/[opt.name]/web/')
        self.parser.add_argument('--update_html_freq', type=int, default=10, help='frequency of saving training results to html')
        self.parser.add_argument('--display_validate_freq', type=int, default=5, help="test validate dateset every $validate_freq forward and backward iteration")
        self.parser.add_argument('--display_train_freq', type=int, default=1, help="print train loss and accuracy every $train_freq forward and backward iteration")
        self.parser.add_argument('--display_data_freq', type=int, default=1, help='frequency of showing training data on screen')

    def parse(self):
        opt = self.parser.parse_args()
        
        # model dir
        opt.model_dir = opt.dir + "/trainer_" + opt.name + "/Train/" 
        opt.data_dir = opt.dir + "/trainer_" + opt.name + "/Data/"
        opt.test_dir = opt.dir + "/trainer_" + opt.name + "/Test/"
        # TODO handle condition when destinate directory exists
        if not os.path.exists(opt.model_dir):
            os.makedirs(opt.model_dir)
        if not os.path.exists(opt.data_dir):
            os.makedirs(opt.data_dir)
        
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

        # mode
        if opt.mode not in ["Train", "Test", "train", "test"]:
            raise Exception("cannot recognize flag `mode`")
        if opt.mode in ['train', "Train"]:
            opt.mode = "Train"
        if opt.mode in ['test', 'Test']:
            opt.mode = "Test"
        
        opt.top_k = eval(opt.top_k)
        opt.mean = eval(opt.mean)
        opt.std = eval(opt.std)
        opt.ratio = eval(opt.ratio)
        if opt.loss_weight == "":
            opt.loss_weight=None
        else:
            opt.loss_weight = torch.FloatTensor(eval(opt.loss_weight))

        args = vars(opt) 
        print "--------Arguments--------"
        for k, v in sorted(args.iteritems()):
            print "%s: %s" %(str(k), str(v))
        print "-----------End-----------"
        
        return opt

if __name__ == "__main__":
    op = Options()
    op.parse()
