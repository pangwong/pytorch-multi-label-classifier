import os
import sys
import json
import random
import collections 
import os.path as osp
from torch.utils.data import DataLoader

from dataset import BaseDataset
sys.path.append('../')
from util.util import rmdir


class MultiLabelDataLoader():
    def __init__(self, opt):
        self.opt = opt
        assert os.path.exists(opt.dir + "/data.txt"), "No data.txt found in specified dir"
        assert os.path.exists(opt.dir + "/label.txt"), "No label.txt found in specified dir"
        
        train_dir = opt.data_dir + "/TrainSet/"
        val_dir = opt.data_dir + "/ValidateSet/"
        test_dir = opt.data_dir + "/TestSet/"
        
        # split data
        if not all([osp.exists(train_dir), osp.exists(val_dir), osp.exists(test_dir)]):
            # rm existing directories
            rmdir(train_dir)
            rmdir(val_dir)
            rmdir(test_dir)

            # split data to Train, Val, Test
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
        
        self.label2id = self._LoadLabel(opt.dir + '/label.txt')
        self.num_classes = [len(labels)-2 for labels in self.label2id]
        
        # load dataset
        if opt.mode == "Train": 
            self.train_set = BaseDataset(self.opt, "TrainSet", self.label2id)
            self.val_set = BaseDataset(self.opt, "ValidateSet", self.label2id)
        else:
            # force batch_size for test to 1
            opt.batch_size = 1
            opt.load_thread = 1
            self.test_set = BaseDataset(test_dir, "TestSet", self.label2id)


    def _WriteDataToFile(self, src_data, dst_dir):
        if not osp.exists(dst_dir):
            os.mkdir(dst_dir)
        with open(dst_dir + "/data.txt", 'w') as d:
            for line in src_data:
                d.write(json.dumps(line, separators=(',',':'))+'\n')


    def _DataLoader(self, dataset):
        # TODO add sampler to balance attributes

        dataloader = DataLoader(
            dataset,
            batch_size=self.opt.batch_size,
            shuffle=self.opt.shuffle,
            num_workers=int(self.opt.load_thread), 
            pin_memory=self.opt.cuda,
            drop_last=False)
        return dataloader


    def _LoadLabel(self, label_file):
        label2id = list()
        with open(label_file) as l:
            label_dict = collections.defaultdict(int)
            new_id = 0 
            for line in l.readlines():
                line = line.strip('\n\r').split(';')
                if len(line) == 3: # attr description
                    if len(label_dict) != 0:
                        label2id.append(label_dict)
                        label_dict = collections.defaultdict(int)
                        new_id = 0
                    label_dict["__name__"] = line[2]
                    label_dict["__attr_id__"] = line[1]
                elif len(line) == 2: # attr value description
                    label_dict[line[0]] = (line[1], new_id)
                    new_id += 1
            if len(label_dict) != 0:
                label2id.append(label_dict)
        return label2id


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
    
    def GetLabel2Id():
        return self.label2id
