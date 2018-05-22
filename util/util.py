import os
import copy
import numpy as np
import logging
import collections
from PIL import Image


def tensor2im(image_tensor, mean, std, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = image_numpy.transpose(1, 2, 0)
    image_numpy *= std
    image_numpy += mean
    image_numpy *= 255.0
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

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
        logging.info("[ %s Accu ] of Test Dataset:" % (label))
    else:
        logging.info("[ %s Accu ] of Epoch %d Batch %d" %(label, epoch, batch_iter))
    
    for index, item in enumerate(accuracy_list):
        for top_k, value in item.iteritems():
            logging.info("----Attribute %d Top%d: %f" %(index, top_k, value["ratio"]))

def opt2file(opt, dst_file):
    args = vars(opt) 
    with open(dst_file, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        print '------------ Options -------------'
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
            print "%s: %s" %(str(k), str(v))
        opt_file.write('-------------- End ----------------\n')
        print '-------------- End ----------------'

def load_label(label_file):
    rid2name = list()   # rid: real id, same as the id in label.txt
    id2rid = list()     # id: number from 0 to len(rids)-1 corresponding to the order of rids
    rid2id = list()     
    with open(label_file) as l:
        rid2name_dict = collections.defaultdict(str)
        id2rid_dict = collections.defaultdict(str)
        rid2id_dict = collections.defaultdict(str)
        new_id = 0 
        for line in l.readlines():
            line = line.strip('\n\r').split(';')
            if len(line) == 3: # attr description
                if len(rid2name_dict) != 0:
                    rid2name.append(rid2name_dict)
                    id2rid.append(id2rid_dict)
                    rid2id.append(rid2id_dict)
                    rid2name_dict = collections.defaultdict(str)
                    id2rid_dict = collections.defaultdict(str)
                    rid2id_dict = collections.defaultdict(str)
                    new_id = 0
                rid2name_dict["__name__"] = line[2]
                rid2name_dict["__attr_id__"] = line[1]
            elif len(line) == 2: # attr value description
                rid2name_dict[line[0]] = line[1]
                id2rid_dict[new_id] = line[0]
                rid2id_dict[line[0]] = new_id
                new_id += 1
        if len(rid2name_dict) != 0:
            rid2name.append(rid2name_dict)
            id2rid.append(id2rid_dict)
            rid2id.append(rid2id_dict)
    return rid2name, id2rid, rid2id

