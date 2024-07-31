from __future__ import print_function
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from types import MethodType
import models
from utils.metric import accuracy, AverageMeter, Timer
import numpy as np
from torch.optim import Optimizer
import contextlib
import os
from .default import NormalNN, weight_reset, accumulate_acc
import copy
import torchvision
from utils.schedulers import CosineSchedule
from torch.autograd import Variable, Function

class Prompt(NormalNN):

    def __init__(self, learner_config):
        self.prompt_param = learner_config['prompt_param']

        super(Prompt, self).__init__(learner_config)

    def update_model(self, inputs, targets,classnames=None):#相当于forward函数
        # logits
        logits, prompt_loss = self.model(inputs, targets,classnames, train=True)
        zeros = torch.zeros((logits.size(0), self.last_valid_out_dim)).cuda()
        logits = torch.cat((zeros, logits), dim=1)
        # logits = logits[:,:self.valid_out_dim]
        # labels = targets-self.last_valid_out_dim#出现数组越界报错的问题就在这里，logits只有当前task的10个class值0-9，而label却会不断增大，二者不匹配
        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets,dw_cls)

        # ce loss
        # total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

    # sets model optimizers
    def init_optimizer(self):
        # parse optimizer args
        # Multi-GPU
        if len(self.config['gpuid']) > 1:#待训练参数包括prompt模块和最后分类层的参数
            # params_to_opt = list(self.model.module.prompt.parameters()) + list(self.model.module.last.parameters())
            for param in self.model.module.parameters():
                param.requires_grad = False
            for param in self.model.module.prompt.parameters():
                param.requires_grad = True
            params_to_opt = list(self.model.module.prompt.parameters())
        else:
            params_to_opt = list(self.model.prompt.parameters()) + list(self.model.last.parameters())
        print('*****************************************')
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        optimizer_arg = {'params':params_to_opt,
                         'lr':self.config['lr'],
                         'weight_decay':self.config['weight_decay']}
        if self.config['optimizer'] in ['SGD','RMSprop']:
            optimizer_arg['momentum'] = self.config['momentum']
        elif self.config['optimizer'] in ['Rprop']:
            optimizer_arg.pop('weight_decay')
        elif self.config['optimizer'] == 'amsgrad':
            optimizer_arg['amsgrad'] = True
            self.config['optimizer'] = 'Adam'
        elif self.config['optimizer'] == 'Adam':
            optimizer_arg['betas'] = (self.config['momentum'],0.999)

        # create optimizers
        self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)#所有的优化器都在这个包下，**optimizer_arg 的含义是将字典 optimizer_arg 中的键值对作为关键字参数传递给 torch.optim 中的优化器构造函数。
        
        # create schedules 学习率调度器
        if self.schedule_type == 'cosine':
            self.scheduler = CosineSchedule(self.optimizer, K=self.schedule[-1])
        elif self.schedule_type == 'decay':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.schedule, gamma=0.1)

    def create_model(self):
        pass

    def cuda(self):
        torch.cuda.set_device(self.config['gpuid'][0])#将当前CUDA设备设置为配置文件中的第一个GPU
        self.model = self.model.cuda()#将模型和损失函数移动到CUDA设备
        self.criterion_fn = self.criterion_fn.cuda()

        # Multi-GPU
        if len(self.config['gpuid']) > 1:#如果gpuid列表中有多个GPU ID（即大于1），则使用torch.nn.DataParallel将模型封装起来，使其能够在多个GPU上并行运行。
            self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
        return self

# Our method!
class CODAPrompt(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt, self).__init__(learner_config)


    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param)
        return model


    def update_model(self, inputs, targets,classnames=None):#相当于forward函数
        # logits
        logits, prompt_loss = self.model(inputs, train=True)
        logits = logits[:,:self.valid_out_dim]
        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets,dw_cls)

        # ce loss
        # total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

class CODAPrompt_text(Prompt):

    def __init__(self, learner_config):
        super(CODAPrompt_text, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        # model_type：zoo；model_name：clip_pt
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'coda',prompt_param=self.prompt_param)
        return model

    def update_model(self, inputs, targets,classnames=None):#相当于forward函数
        # logits
        logits, prompt_loss = self.model(inputs, targets,classnames, train=True)
        zeros = torch.zeros((logits.size(0), self.last_valid_out_dim)).cuda()
        logits = torch.cat((zeros, logits), dim=1)
        # logits = logits[:,:self.valid_out_dim]
        # labels = targets-self.last_valid_out_dim#出现数组越界报错的问题就在这里，logits只有当前task的10个class值0-9，而label却会不断增大，二者不匹配
        # ce with heuristic
        logits[:,:self.last_valid_out_dim] = -float('inf')
        dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
        total_loss = self.criterion(logits, targets,dw_cls)

        # ce loss
        # total_loss = total_loss + prompt_loss.sum()

        # step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.detach(), logits

class DualPrompt(Prompt):

    def __init__(self, learner_config):
        super(DualPrompt, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'dual', prompt_param=self.prompt_param)
        return model

# @inproceedings{wang2022learning,
#   title={Learning to prompt for continual learning},
#   author={Wang, Zifeng and Zhang, Zizhao and Lee, Chen-Yu and Zhang, Han and Sun, Ruoxi and Ren, Xiaoqi and Su, Guolong and Perot, Vincent and Dy, Jennifer and Pfister, Tomas},
#   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
#   pages={139--149},
#   year={2022}
# }
class L2P(Prompt):

    def __init__(self, learner_config):
        super(L2P, self).__init__(learner_config)

    def create_model(self):
        cfg = self.config
        model = models.__dict__[cfg['model_type']].__dict__[cfg['model_name']](out_dim=self.out_dim, prompt_flag = 'l2p',prompt_param=self.prompt_param)
        return model