import os
import sys
import json
import random
import logging
import torch.utils.data as data

from .transformer import get_transformer, load_image

class BaseDataset(data.Dataset):
    def __init__(self, opt, data_type, id2rid):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.data_type = data_type
        self.dataset = self._load_data(opt.data_dir+ '/' + data_type + '/data.txt')
        self.id2rid = id2rid
        self.data_size = len(self.dataset)
        self.transformer = get_transformer(opt)

    def __getitem__(self, index):
        image_file, box, attr_ids = self.dataset[index % self.data_size]
        
        input = load_image(image_file, box, self.opt, self.transformer)

        # label
        labels = list()
        for index, attr_id in enumerate(attr_ids):
            labels.append(self.id2rid[index][attr_id])

        return input, labels

    def __len__(self):
        return self.data_size

    def _load_data(self, data_file):
        dataset = list()
        if not os.path.exists(data_file):
            return dataset
        with open(data_file) as d:
            for line in d.readlines():
                line = json.loads(line)
                dataset.append(self.readline(line))
        if self.opt.shuffle:
            logging.info("Shuffle %s Data" %(self.data_type))
            random.shuffle(dataset)
        else:
            logging.info("Not Shuffle %s Data" %(self.data_type))
        return dataset
    
    def readline(self, line):
        data = [None, None, None]
        if line.has_key('image_file'):
            data[0] = line["image_file"]
        if line.has_key('box'):
            data[1] = line["box"]
        if line.has_key('id'):
            data[2] = line["id"]
        return data
