import torch.utils.data as data
import torchvision.transforms as transforms
import json
import os
import os.path as osp
import cv2
import copy
import random
from PIL import Image
import matplotlib.pyplot as plt

class BaseDataset(data.Dataset):
    def __init__(self, opt, data_type, label2id):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.data_type = data_type
        self.dataset = self._load_data(opt.data_dir+ '/' + data_type + '/data.txt')
        self.label2id = label2id
        self.data_size = len(self.dataset)
        self.transformer = self._get_transformer()

    def __getitem__(self, index):
        image_file, box, attr_labels = self.dataset[index % self.data_size]
        img = Image.open(image_file)
        if self.opt.input_channel == 3:
            img = img.convert('RGB')
        
        width, height = img.size
        
        # box crop
        if box is not None and self.opt.region == True:
            box = self._fix_box(box, width, height, self.opt.box_ratio, self.opt.box_scale)
            area = (box['x'], box['y'], box['x']+box['w'], box['y']+box['h'])
            img = img.crop(area)
        
        # transform
        input = self.transformer(img)

        # label
        labels = list()
        for index, attr_label in enumerate(attr_labels):
            labels.append(self.label2id[index][attr_label][1])

        return input, labels

    def __len__(self):
        return self.data_size

    def _load_data(self, data_file):
        dataset = list()
        if not osp.exists(data_file):
            return dataset
        with open(data_file) as d:
            for line in d.readlines():
                line = json.loads(line)
                dataset.append(self._read(line))
        if self.opt.shuffle:
            print "Shuffle %s Data" %(self.data_type)
            random.shuffle(dataset)
        return dataset
    
    def _get_transformer(self):
        transform_list = []
        
        # resize  
        osize = [self.opt.load_size, self.opt.load_size]
        transform_list.append(transforms.Resize(osize, Image.BICUBIC))
        
        # grayscale
        if self.opt.input_channel == 1:
            transform_list.append(transforms.Grayscale())

        # crop
        if self.opt.crop == "RandomCrop":
            transform_list.append(transforms.RandomCrop(self.opt.fineSize))
        elif self.opt.crop == "CenterCrop":
            transform_list.append(transforms.CenterCrop(self.opt.input_size))
        elif self.opt.crop == "FiveCrop":
            transform_list.append(transforms.FiveCrop(self.opt.input_size))
        elif self.opt.crop == "TenCrop":
            transform_list.append(transforms.TenCrop(self.opt.input_size))
        
        # flip
        if self.opt.mode == "Train" and self.opt.flip:
            transform_list.append(transforms.RandomHorizontalFlip())

        # to tensor
        transform_list.append(transforms.ToTensor())
        
        # If you make changes here, you should also modified 
        # function `tensor2im` in util/util.py accordingly
        transform_list.append(transforms.Normalize(self.opt.mean, self.opt.std))

        return transforms.Compose(transform_list)

    def _read(self, line):
        data = [None, None, None]
        if line.has_key('image_file'):
            data[0] = line["image_file"]
        if line.has_key('box'):
            data[1] = line["box"]
        if line.has_key('id'):
            data[2] = line["id"]
        return data


    def _fix_box(self, box, width, height, ratio=-1, scale=1.0):
        box = copy.deepcopy(box)
        w = box["w"]
        h = box["h"]
        x = box["x"] + w / 2
        y = box["y"] + h / 2
        mw = 2 * min(x, width - x)
        mh = 2 * min(y, height - y)
        w = max(1, min(int(w * scale), mw))
        h = max(1, min(int(h * scale), mh))
        if ratio > 0:
          if 1.0 * w / h > ratio:
              h = int(w / ratio)
              h = min(h, mh)
              w = int(h * ratio)
          else:
              w = int(h * ratio)
              w = min(w, mw)
              h = int(w / ratio)
        box["x"] = x - w / 2
        box["y"] = y - h / 2
        box["w"] = w
        box["h"] = h
        return box

