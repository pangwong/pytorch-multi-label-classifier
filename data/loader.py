import os
import sys
import json
import random
import logging
import collections 
from torch.utils.data import DataLoader

from dataset import BaseDataset
sys.path.append('../')
from util.util import rmdir, load_label

class MultiLabelDataLoader():
    def __init__(self, opt):
        self.opt = opt
        assert os.path.exists(opt.dir + "/data.txt"), "No data.txt found in specified dir"
        assert os.path.exists(opt.dir + "/label.txt"), "No label.txt found in specified dir"
        
        train_dir = opt.data_dir + "/TrainSet/"
        val_dir = opt.data_dir + "/ValidateSet/"
        test_dir = opt.data_dir + "/TestSet/"
         
        # split data
        if not all([os.path.exists(train_dir), os.path.exists(val_dir), os.path.exists(test_dir)]):
            # rm existing directories
            rmdir(train_dir)
            rmdir(val_dir)
            rmdir(test_dir)

            # split data to Train, Val, Test
            logging.info("Split raw data to Train, Val and Test")
            ratios = opt.ratio
            dataset = collections.defaultdict(list)
            with open(opt.dir + '/data.txt') as d:
                for line in d.readlines():
                    line = json.loads(line)
                    # if data has been specified data_type yet, load data as what was specified before
                    if line.has_key("type"):
                        dataset[line["type"]].append(line)
                        continue
                    # specified data_type randomly
                    rand = random.random()
                    if rand < ratios[0]:
                        data_type = "Train"
                    elif rand < ratios[0] + ratios[1]:
                        data_type = "Validate"
                    else:
                        data_type = "Test"
                    dataset[data_type].append(line)
            # write to file
            self._WriteDataToFile(dataset["Train"], train_dir)
            self._WriteDataToFile(dataset["Validate"], val_dir)
            self._WriteDataToFile(dataset["Test"], test_dir)
        
        self.rid2name, self.id2rid, self.rid2id = load_label(opt.dir + '/label.txt')
        self.num_classes = [len(item)-2 for item in self.rid2name]
        
        # load dataset
        if opt.mode == "Train": 
            logging.info("Load Train Dataset...")
            self.train_set = BaseDataset(self.opt, "TrainSet", self.rid2id)
            logging.info("Load Validate Dataset...")
            self.val_set = BaseDataset(self.opt, "ValidateSet", self.rid2id)
        else:
            # force batch_size for test to 1
            self.opt.batch_size = 1
            self.opt.load_thread = 1
            logging.info("Load Test Dataset...")
            self.test_set = BaseDataset(self.opt, "TestSet", self.rid2id)

    def GetTrainSet(self):
        if self.opt.mode == "Train":
            return self._DataLoader(self.train_set)
        else:
            raise("Train Set DataLoader NOT implemented in Test Mode")

    def GetValSet(self):
        if self.opt.mode == "Train":
            return self._DataLoader(self.val_set)
        else:
            raise("Validation Set DataLoader NOT implemented in Test Mode")

    def GetTestSet(self):
        if self.opt.mode == "Test":
            return self._DataLoader(self.test_set)
        else:
            raise("Test Set DataLoader NOT implemented in Train Mode")
    
    def GetNumClasses(self):
        return self.num_classes
    
    def GetRID2Name(self):
        return self.rid2name
    
    def GetID2RID(self):
        return self.id2rid
    
    def GetiRID2ID(self):
        return self.irid2id

    def _WriteDataToFile(self, src_data, dst_dir):
        """
            write info of each objects to data.txt as predefined format
        """
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        with open(dst_dir + "/data.txt", 'w') as d:
            for line in src_data:
                d.write(json.dumps(line, separators=(',',':'))+'\n')


    def _DataLoader(self, dataset):
        """
            create data loder
        """
        dataloader = DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=self.opt.shuffle,
            num_workers=int(self.opt.load_thread), 
            pin_memory=self.opt.cuda,
            drop_last=False)
        return dataloader

