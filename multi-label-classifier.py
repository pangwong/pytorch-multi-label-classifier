import os
import time
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from collections import OrderedDict, defaultdict

from data.loader import MultiLabelDataLoader
from options.options import Options
from models.build_model import BuildMultiLabelModel
from models.lightcnn import LightCNN_29Layers_v2_templet 
from models.alexnet import AlexNetTemplet
from util import util 
from util.webvisualizer import WebVisualizer


def _forward(model, criterion, inputs, targets, opt, phase):
    if opt.cuda:
        inputs = inputs.cuda(opt.devices[0], async=True)

    if phase == "Train":
        inputs_var = Variable(inputs, requires_grad=True)
        model.train()
    elif phase in ["Validate", "Test"]:
        inputs_var = Variable(inputs, volatile=True)
        model.eval()
        
    # forward
    if opt.cuda:
        if len(opt.devices) > 1:
            output = nn.parallel.data_parallel(model, inputs_var, opt.devices)
        else:
            output = model(inputs_var)
    else:
        output = model(inputs_var)
    
    # calculate loss 
    target_vars = list()
    for index in range(len(targets)):
        if opt.cuda:
            targets[index] = targets[index].cuda(opt.devices[0], async=True)
        target_vars.append(Variable(targets[index]))
    loss_list = list()
    loss = Variable(torch.FloatTensor(1)).zero_()
    if opt.cuda:
        loss = loss.cuda(opt.devices[0])
    for index in range(len(targets)):
        sub_loss = criterion(output[index], target_vars[index])
        loss_list.append(sub_loss.data[0])
        loss += sub_loss
    
    return output, loss, loss_list


def _accuracy(outputs, targets, score_thres, top_k=(1,)):
    max_k = max(top_k)
    accuracy = []
    thres_list = eval(score_thres)
    if isinstance(thres_list, float) or isinstance(thres_list, int) :
        thres_list = [eval(score_thres)]*len(targets)

    for i in range(len(targets)):
        target = targets[i]
        output = outputs[i].data
        batch_size = output.size(0)
        curr_k = min(max_k, output.size(1))
        top_value, index = output.cpu().topk(curr_k, 1)
        index = index.t()
        top_value = top_value.t()
        correct = index.eq(target.cpu().view(1,-1).expand_as(index))
        mask = (top_value>=thres_list[i])
        correct = correct*(mask)
        #print "masked correct: ", correct
        res = defaultdict(dict)
        for k in top_k:
            k = min(k, output.size(1))
            correct_k = correct[:k].view(-1).float().sum(0)[0]
            res[k]["s"] = batch_size
            res[k]["r"] = correct_k
            res[k]["ratio"] = float(correct_k)/batch_size
        accuracy.append(res)
    return accuracy


def _forward_dataset(model, criterion, data_loader, opt):
    sum_batch = 0 
    accuracy = list()
    avg_loss = list()
    for i, data in enumerate(data_loader):
        if opt.mode == "Train":
            if random.random() > opt.validate_ratio:
                continue
        if opt.mode == "Test":
            print "test %s/%s image" %(i, len(data_loader))
        sum_batch += 1
        inputs, targets = data
        output, loss, loss_list = _forward(model, criterion, inputs, targets, opt, "Validate")
        batch_accuracy = _accuracy(output, targets, opt.score_thres, opt.top_k) 
        if len(accuracy) == 0:
            accuracy = copy.deepcopy(batch_accuracy)
            for index, item in enumerate(batch_accuracy):
                for k,v in item.iteritems():
                    accuracy[index][k]["ratio"] = v["ratio"]
            continue
        for index, item in enumerate(batch_accuracy):
            for k,v in item.iteritems():
                accuracy[index][k]["ratio"] += v["ratio"]
        for item in loss_list:
            if len(avg_loss) == 0:
                avg_loss = copy.deepcopy(loss_list) 
            else:
                for index, loss in enumerate(loss_list):
                    avg_loss[index] += loss
    # average on batches
    for index, item in enumerate(accuracy):
        for k,v in item.iteritems():
            accuracy[index][k]["ratio"] /= float(sum_batch)
    for index in range(len(avg_loss)):
        avg_loss[index] /= float(sum_batch)

    return accuracy, avg_loss

def validate(model, criterion, val_set, opt):
    return _forward_dataset(model, criterion, val_set, opt)

def test(model, criterion, test_set, opt):
    print "####################Test Model###################"
    test_accuracy, test_loss = _forward_dataset(model, criterion, test_set, opt)
    print "data_dir:   ", opt.data_dir + "/TestSet/"
    print "state_dict: ", opt.model_dir + "/" + opt.checkpoint_name
    util.print_loss(test_loss, "Test")
    util.print_accuracy(test_accuracy, "Test")
    test_result = os.path.join(opt.test_dir, "result.txt")
    with open(test_result, 'w') as t:
        for index, item in enumerate(test_accuracy):
            t.write("Attribute %d:\n" %(index))
            for top_k, value in item.iteritems():
                t.write("----Accuracy of Top%d: %f\n" %(top_k, value["ratio"]))  
    print "#################Finished Testing################"

def train(model, criterion, train_set, val_set, opt):
    # define web visualizer using visdom
    webvis = WebVisualizer(opt)
    
    # modify learning rate of last layer
    finetune_params = _modify_last_layer_lr(model.named_parameters(), opt.lr, opt.lr_mult_w, opt.lr_mult_b)
    # define optimizer
    optimizer = optim.SGD(finetune_params, 
                          opt.lr, 
                          momentum=opt.momentum, 
                          weight_decay=opt.weight_decay)
    # define laerning rate scheluer
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=opt.lr_decay_in_epoch,
                                          gamma=opt.gamma)

    # record forward and backward times 
    train_batch_num = len(train_set)
    total_batch_iter = 0
    print "####################Train Model###################"
    for epoch in range(opt.sum_epoch):
        epoch_start_t = time.time()
        epoch_batch_iter = 0
        print 'Begin of epoch %d' %(epoch)
        for i, data in enumerate(train_set):
            iter_start_t = time.time()
            # train 
            inputs, targets = data
            output, loss, loss_list = _forward(model, criterion, inputs, targets, opt, "Train")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            webvis.reset()
            epoch_batch_iter += 1
            total_batch_iter += 1

            # display train loss and accuracy
            if total_batch_iter % opt.display_train_freq == 0:
                # accuracy
                batch_accuracy = _accuracy(output, targets, opt.score_thres, opt.top_k) 
                util.print_loss(loss_list, "Train", epoch, total_batch_iter)
                util.print_accuracy(batch_accuracy, "Train", epoch, total_batch_iter)
                if opt.display_id > 0:
                    x_axis = epoch + float(epoch_batch_iter)/train_batch_num
                    # TODO support accuracy visualization of multiple top_k
                    plot_accuracy = [batch_accuracy[i][opt.top_k[0]] for i in range(len(batch_accuracy)) ]
                    accuracy_list = [item["ratio"] for item in plot_accuracy]
                    webvis.plot_points(x_axis, loss_list, "Loss", "Train")
                    webvis.plot_points(x_axis, accuracy_list, "Accuracy", "Train")
            
            # display train data 
            if total_batch_iter % opt.display_data_freq == 0:
                image_list = list()
                for index in range(inputs.size()[0]): 
                    input_im = util.tensor2im(inputs[index])
                    image_list.append(("Input_"+str(index), input_im))
                image_dict = OrderedDict(image_list)
                save_result = total_batch_iter % opt.update_html_freq
                #webvis.plot_images(image_dict, opt.display_id + 2*len(num_classes), epoch, save_result)
            
            # validate and display validate loss and accuracy
            if total_batch_iter % opt.display_validate_freq == 0 and len(val_set) > 0:
                val_accuracy, val_loss = validate(model, criterion, val_set, opt)
                x_axis = epoch + float(epoch_batch_iter)/train_batch_num
                accuracy_list = [val_accuracy[i][opt.top_k[0]]["ratio"] for i in range(len(val_accuracy))]
                util.print_loss(val_loss, "Validate", epoch, total_batch_iter)
                util.print_accuracy(val_accuracy, "Validate", epoch, total_batch_iter)
                if opt.display_id > 0:
                    webvis.plot_points(x_axis, val_loss, "Loss", "Validate")
                    webvis.plot_points(x_axis, accuracy_list, "Accuracy", "Validate")

            # save snapshot 
            if total_batch_iter % opt.save_batch_iter_freq == 0:
                print "saving the latest model (epoch %d, total_batch_iter %d)" %(epoch, total_batch_iter)
                util.save_model(model, opt, epoch)
                # TODO snapshot loss and accuracy
            
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.sum_epoch, time.time() - epoch_start_t))
        
        if epoch % opt.save_epoch_freq:
            print 'saving the model at the end of epoch %d, iters %d' %(epoch, total_batch_iter)
            util.save_model(model, opt, epoch) 

        # adjust learning rate 
        scheduler.step()
        lr = optimizer.param_groups[0]['lr'] 
        print('learning rate = %.7f' % lr, 'epoch = %d' %(epoch)) 

def _modify_last_layer_lr(named_params, base_lr, lr_mult_w, lr_mult_b):
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
    

def main():
    # parse option 
    op = Options()
    opt = op.parse()

    # load train or test data
    data_loader = MultiLabelDataLoader(opt)
    if opt.mode == "Train":
        train_set = data_loader.GetTrainSet()
        val_set = data_loader.GetValSet()
    elif opt.mode == "Test":
        test_set = data_loader.GetTestSet()

    num_classes = data_loader.GetNumClasses()

    # load model
    templet = LightCNN_29Layers_v2_templet(opt.input_channel) 
    #templet = AlexNetTemplet(opt.input_channel)
    tmp_input = Variable(torch.FloatTensor(1, opt.input_channel, opt.input_size, opt.input_size))
    tmp_output = templet(tmp_input)
    output_dim = int(tmp_output.size()[-1])
    model = BuildMultiLabelModel(templet, output_dim, num_classes)
    print model
    
    # load exsiting model
    if opt.checkpoint_name != "":
        if os.path.exists(opt.checkpoint_name):
            model.load_state_dict(torch.load(opt.checkpoint_name))
        else:
            model.load_state_dict(torch.load(opt.model_dir + "/" + opt.checkpoint_name))

    # define loss function
    criterion = nn.CrossEntropyLoss(weight=opt.loss_weight) 

    if opt.cuda:
        model = model.cuda(opt.devices[0])
        criterion = criterion.cuda(opt.devices[0])
        cudnn.benchmark = True
    
    if opt.mode == "Train":
        # Train model
        train(model, criterion, train_set, val_set, opt)
    elif opt.mode == "Test":
        # Test model
        test(model, criterion, test_set, opt)


if __name__ == "__main__":
    main()
