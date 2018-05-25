import os
import sys
import json
import logging
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict, defaultdict

from options.options import Options
from models.model import load_model
from data.transformer import  get_transformer, load_image
from util.util import load_label, opt2file
from util.webvisualizer import WebVisualizer

def main():
    # parse options 
    op = Options()
    opt = op.parse()

    # special setting
    opt.shuffle = False
    opt.batch_size = 1
    opt.load_thread = 1

    # initialize train or test working dir
    test_dir = os.path.join(opt.classify_dir , opt.name)
    opt.model_dir = opt.dir + "/trainer_" + opt.name + "/Train/"
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    # save options to disk
    opt2file(opt, os.path.join(test_dir, "opt.txt"))
    
    # log setting 
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    fh = logging.FileHandler(test_dir + "/deploy.log", 'a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    logging.getLogger().addHandler(ch)
    logging.getLogger().setLevel(logging.INFO)
    
    # load label  
    if opt.label_file == "":
        opt.label_file = opt.dir + "/label.txt"
    rid2name, id2rid, rid2id = load_label(opt.label_file)
    num_classes = [len(rid2name[index])-2 for index in range(len(rid2name))]
        
    # load transformer
    transformer = get_transformer(opt) 

    # load model
    model = load_model(opt, num_classes)
    model.eval()
    
    # use cuda
    if opt.cuda:
        model = model.cuda(opt.devices[0])
        cudnn.benchmark = True
    
    l = open(test_dir + "/classify_res_data.txt", 'w')
    with open(opt.classify_dir + "/data.txt") as data:
        for num, line in enumerate(data):
            logging.info(str(num+1))
            line = json.loads(line)
            input_tensor = load_image(line["image_file"], line["box"], opt, transformer) 
            input_tensor = input_tensor.unsqueeze(0)
            if opt.cuda:
                input_tensor = input_tensor.cuda(opt.devices[0])
            outputs = model(Variable(input_tensor, volatile=True)) 
            if not isinstance(outputs, list):
                outputs = [outputs]
            line["classify_res"] = list() 
            for index, out in enumerate(outputs):
                out = out.cpu()
                #print "out:", out
                softmax = F.softmax(out, dim=1).data.squeeze()
                #print "softmax:", softmax 
                probs, ids = softmax.sort(0, True)
                classify_res = {}
                for i in range(len(probs)):
                    classify_res[rid2name[index][id2rid[index][ids[i]]]] = probs[i]
                classify_res["max_score"] = probs[0]
                classify_res["best_label"] = rid2name[index][id2rid[index][ids[0]]]
                line["classify_res"].append(classify_res)
            l.write(json.dumps(line, separators=(',', ':'))+'\n')
    l.close()
    logging.info("classification done")


if __name__ == "__main__":
    main()
