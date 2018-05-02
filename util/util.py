import os
import torch
import numpy as np
import logging
from PIL import Image

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def save_model(model, opt, epoch):
    checkpoint_name = opt.model_dir + "/epoch_%s_snapshot.pth" %(epoch)
    torch.save(model.cpu().state_dict(), checkpoint_name)
    if opt.cuda and torch.cuda.is_available():
        model.cuda(opt.devices[0])

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rmdir(path):
    if os.path.exists(path):
        os.system('rm -rf ' + path)

def print_loss(loss_list, label, epoch=0, batch_iter=0):
    if label == "Test":
        logging.info("[ %s Loss ] of Test Dataset:" % (label))
    else:
        logging.info("[ %s Loss ] of Epoch %d Batch %d" % (label, epoch, batch_iter))
    
    for index, loss in enumerate(loss_list):
        logging.info("----Attribute %d:  %f" %(index, loss))

def print_accuracy(accuracy_list, label, epoch=0, batch_iter=0):
    if label == "Test":
        logging.info("[ %s Accuracy ] of Test Dataset:" % (label))
    else:
        logging.info("[ %s Accuracy ] of Epoch %d Batch %d" %(label, epoch, batch_iter))
    
    for index, item in enumerate(accuracy_list):
        for top_k, value in item.iteritems():
            logging.info("----Attribute %d Top%d: %f" %(index, top_k, value["ratio"]))
