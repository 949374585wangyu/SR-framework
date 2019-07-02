# -*- coding:utf-8 -*- 
'''
 * @Author: wjm 
 * @Date: 2019-06-14 11:27:29 
 * @Last Modified by:   wjm 
 * @Last Modified time: 2019-06-14 11:27:29 
 * @Desc: 
'''
from __future__ import print_function
from option import opt
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Net import Net as Srnet
from data import get_training_set, get_eval_set
import pdb
import socket
import time
from utils import *
from base_network import *
from math import log10

gpus_list = range(opt.gpus)
cudnn.benchmark = True
if not os.path.exists('log'):
    os.mkdir('log')
save_config(opt)

def train(epoch):

    criterion = myloss(opt)
    epoch_loss = 0
    if cuda:
        criterion = criterion.cuda(gpus_list[0])
    
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, bicubic = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        optimizer.zero_grad()
        t0 = time.time()
        prediction = model(input)

        if opt.residual:
            prediction = prediction + bicubic

        loss = criterion(prediction, target)
        t1 = time.time()
        epoch_loss += loss.data
        loss.backward()
        optimizer.step()
        log = "===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.data, (t1 - t0))
        write_log(log, refresh=True)

    log = "===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader))
    write_log(log, refresh=True)

def test():
    avg_psnr = 0
    criterion = nn.MSELoss()
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if cuda:
            input = input.cuda(gpus_list[0])
            target = target.cuda(gpus_list[0])

        prediction = model(input)
        mse = criterion(prediction, target)
        psnr = 10 * log10(1 / mse.data[0])
        avg_psnr += psnr
    log = "===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader))
    write_log(log)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('Loading datasets!')
train_set = get_training_set(
        opt.data_dir, 
        opt.hr_train_dataset, 
        opt.upscale_factor, 
        opt.patch_size, 
        opt.data_augmentation
    )
training_data_loader = DataLoader(
        dataset=train_set, 
        num_workers=opt.threads, 
        batch_size=opt.batchSize, 
        shuffle=True
    )    
test_set = get_eval_set(os.path.join(opt.data_dir,opt.hr_valid_dataset), opt.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)

write_log('Buildind model!')
model = model = Srnet(
        num_channels=opt.n_colors, 
        base_filter=opt.base_filter,  
        num_stages=opt.num_stages, 
        scale_factor=opt.upscale_factor
    ) 

model = torch.nn.DataParallel(model, device_ids=gpus_list)

write_log('---------- Networks architecture -------------')
print('Saving config!')
write_log(model, refresh=True)
write_log('----------------------------------------------')

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        write_log('Pre-trained SR model is loaded.') 

if cuda:
    model = model.cuda(gpus_list[0])

optimizer = make_optim(opt, model.parameters())

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch)
    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch+1) % (opt.nEpochs/2) == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        log = 'Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr'])
        write_log(log)
            
    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(opt, epoch, model)
        if opt.valid:
            test()