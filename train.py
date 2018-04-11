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
    for i in range(len(targets)):
        target = targets[i]
        output = outputs[i].data
        batch_size = output.size(0)
        curr_k = min(max_k, output.size(1))
        top_value, index = output.cpu().topk(curr_k, 1)
        index = index.t()
        top_value = top_value.t()
        correct = index.eq(target.cpu().view(1,-1).expand_as(index))
        mask = (top_value>=score_thres)
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


def validate(model, criterion, val_set, opt):
    sum_batch = 0 
    accuracy = list()
    avg_loss = list()
    for i, data in enumerate(val_set):
        if random.random() > opt.validate_ratio:
            continue
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
    
def test(model, test_loader, opt):
    pass


def main():
    # parse option 
    op = Options()
    opt = op.parse()

    # define web visualizer using visdom
    webvis = WebVisualizer(opt)

    # load train data
    opt.mode == "Train"
    data_loader = MultiLabelDataLoader(opt)
    train_set = data_loader.GetTrainSet()
    val_set = data_loader.GetValSet()
    num_classes = data_loader.GetNumClasses()
    batch_number = len(train_set)

    # load model
    templet = LightCNN_29Layers_v2_templet(opt.input_channel) 
    #templet = AlexNetTemplet(opt.input_channel)
    tmp_input = Variable(torch.FloatTensor(1, opt.input_channel, opt.input_size, opt.input_size))
    tmp_output = templet(tmp_input)
    output_dim = int(tmp_output.size()[-1])
    model = BuildMultiLabelModel(templet, output_dim, num_classes)
    # TODO load state_dict from disk
    model.train()
    print model

    # define loss
    criterion = nn.CrossEntropyLoss(weight=opt.loss_weight) 

    if opt.cuda:
        model = model.cuda(opt.devices[0])
        criterion = criterion.cuda(opt.devices[0])
        cudnn.benchmark = True

    # define optimizer
    optimizer = optim.SGD(model.parameters(), 
                          opt.lr, 
                          momentum=opt.momentum, 
                          weight_decay=opt.weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size=opt.lr_decay_in_epoch,
                                          gamma=opt.gamma)
    total_batch_iter = 0
    # record forward and backward times 
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
                    x_axis = epoch + float(epoch_batch_iter)/batch_number
                    # TODO support multiple top_k
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
            if total_batch_iter % opt.display_validate_freq == 0:
                val_accuracy, val_loss = validate(model, criterion, val_set, opt)
                x_axis = epoch + float(epoch_batch_iter)/batch_number
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
        
if __name__ == "__main__":
    main()
