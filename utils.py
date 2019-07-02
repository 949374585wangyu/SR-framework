# -*- coding:utf-8 -*- 
'''
 * @Author: wjm 
 * @Date: 2019-06-14 11:37:40 
 * @Last Modified by:   wjm 
 * @Last Modified time: 2019-06-14 11:37:40 
 * @Desc: 
'''
import os
import time
import datetime
import torch

def get_path(subdir):
    return os.path.join(subdir)

def save_config(args):
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    open_type = 'a' if os.path.exists(get_path('.\log\config.txt'))else 'w'
    with open(get_path('.\log\config.txt'), open_type) as f:
        f.write(now + '\n\n')
        for arg in vars(args):
            f.write('{}: {}\n'.format(arg, getattr(args, arg)))
        f.write('\n')

def write_log(log, refresh=False):
    print(log)
#     now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    open_type = 'a' if os.path.exists(get_path('.\log\log.txt'))else 'w'
    log_file = open(get_path('.\log\log.txt'), open_type)
#     log_file.write(now + '\n\n')
    log_file.write(str(log) + '\n')
    if refresh:
        log_file.close()
        log_file = open(get_path('.\log\log.txt'), 'a')

def checkpoint(opt, epoch, model):
    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
    model_out_path = opt.save_folder+'/'+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))