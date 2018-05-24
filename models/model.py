import os
import torch
import logging
from torch.autograd import Variable

from .build_model import BuildMultiLabelModel, LoadPretrainedModel
from .lightcnn import LightCNN_29Layers_v2_templet, LightCNN_9Layers_templet 
from .alexnet import AlexnetTemplet
from .resnet import Resnet18Templet
from .vgg import VGG16Templet

def load_model(opt, num_classes):
    # load templet
    if opt.model == "Alexnet":
        templet = AlexnetTemplet(opt.input_channel, opt.pretrain)
    elif opt.model == "LightenB":
        templet = LightCNN_29Layers_v2_templet(opt.input_channel, opt.pretrain)
    elif opt.model == "Lighten9":
        templet = LightCNN_9Layers_templet(opt.input_channel, opt.pretrain)
    elif opt.model == "Resnet18":
        templet = Resnet18Templet(opt.input_channel, opt.pretrain)
    elif opt.model == "VGG16":
        templet = VGG16Templet(opt.input_channel, opt.pretrain)
    else:
        logging.error("unknown model type")
        sys.exit(0)
    
    # build model
    tmp_input = Variable(torch.FloatTensor(1, opt.input_channel, opt.input_size, opt.input_size))
    tmp_output = templet(tmp_input)
    output_dim = int(tmp_output.size()[-1])
    model = BuildMultiLabelModel(templet, output_dim, num_classes)
    logging.info(model)
    
    # imagenet pretrain model
    if opt.pretrain:
        logging.info("use imagenet pretrained model")
    
    # load exsiting model
    if opt.checkpoint_name != "":
        if os.path.exists(opt.checkpoint_name):
            logging.info("load pretrained model from "+opt.checkpoint_name)
            model.load_state_dict(torch.load(opt.checkpoint_name))
        elif os.path.exists(opt.model_dir):
            checkpoint_name = opt.model_dir + "/" + opt.checkpoint_name
            model.load_state_dict(torch.load(checkpoint_name))
            logging.info("load pretrained model from "+ checkpoint_name)
        else:
            opt.checkpoint_name = ""
            logging.warning("WARNING: unknown pretrained model, skip it.")

    return model

def save_model(model, opt, epoch):
    checkpoint_name = opt.model_dir + "/epoch_%s_snapshot.pth" %(epoch)
    torch.save(model.cpu().state_dict(), checkpoint_name)
    if opt.cuda and torch.cuda.is_available():
        model.cuda(opt.devices[0])

def modify_last_layer_lr(named_params, base_lr, lr_mult_w, lr_mult_b):
    params = list()
    for name, param in named_params: 
        if 'bias' in name:
            if 'FullyConnectedLayer_' in name:
                params += [{'params':param, 'lr': base_lr * lr_mult_b, 'weight_decay': 0}]
            else:
                params += [{'params':param, 'lr': base_lr * 2, 'weight_decay': 0}]
        else:
            if 'FullyConnectedLayer_' in name:
                params += [{'params':param, 'lr': base_lr * lr_mult_w}]
            else:
                params += [{'params':param, 'lr': base_lr * 1}]
    return params
