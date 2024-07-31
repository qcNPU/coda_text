from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import torch
import numpy as np
import yaml
import json
import random
from trainer import Trainer

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0,1,2,3],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/10-task",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--learner_type', type=str, default='prompt', help="The type (filename) of learner")
    # parser.add_argument('--learner_name', type=str, default='CODAPrompt', help="The class name of learner")
    parser.add_argument('--learner_name', type=str, default='CODAPrompt_text', help="The class name of learner")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[100, 8, 0.0],
                         help="e prompt pool size, e prompt length, g prompt length")

    # Config Arg
    parser.add_argument('--config', type=str, default="configs/cifar-100_prompt.yaml",
                         help="yaml experiment config input")
    parser.add_argument('--dataset', type=str, default="CIFAR100",
                        help="yaml experiment config input")

    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    # 读取配置文件
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))#一部分使用代码里配置的，一部分使用配置文件里配的
    return argparse.Namespace(**config)#`args` 就是一个 argparse.Namespace 对象

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])#它包含了命令行参数。这个列表的第一个元素是脚本的名称，后续的元素是脚本的命令行参数。

    # determinstic backend 用于控制 cuDNN（NVIDIA 的深度神经网络库）中的某些操作是否应该以确定性的方式执行。设置 torch.backends.cudnn.deterministic = True 可以确保某些操作（例如卷积）在多次运行时产生相同的结果，从而提高训练过程的可重复性。
    torch.backends.cudnn.deterministic=True

    # redirect output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)#将 sys.stdout 重定向到自定义的 Logger，用于将程序的标准输出（即 print 语句的输出）重定向到一个文件或其他输出流

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)
    
    metric_keys = ['acc','time',]
    save_keys = ['global', 'pt', 'pt-local']
    global_only = ['time']
    avg_metrics = {}
    for mkey in metric_keys: 
        avg_metrics[mkey] = {}
        for skey in save_keys: avg_metrics[mkey][skey] = []

    # load results
    if args.overwrite:
        start_r = 0
    else:
        try:
            for mkey in metric_keys: 
                for skey in save_keys:
                    if (not (mkey in global_only)) or (skey == 'global'):
                        save_file = args.log_dir+'/results-'+mkey+'/'+skey+'.yaml'
                        if os.path.exists(save_file):
                            with open(save_file, 'r') as yaml_file:
                                yaml_result = yaml.safe_load(yaml_file)
                                avg_metrics[mkey][skey] = np.asarray(yaml_result['history'])

            # next repeat needed
            start_r = avg_metrics[metric_keys[0]][save_keys[0]].shape[-1]

            # extend if more repeats left
            if start_r < args.repeat:
                max_task = avg_metrics['acc']['global'].shape[0]
                for mkey in metric_keys: 
                    avg_metrics[mkey]['global'] = np.append(avg_metrics[mkey]['global'], np.zeros((max_task,args.repeat-start_r)), axis=-1)
                    if (not (mkey in global_only)):
                        avg_metrics[mkey]['pt'] = np.append(avg_metrics[mkey]['pt'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)
                        avg_metrics[mkey]['pt-local'] = np.append(avg_metrics[mkey]['pt-local'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)

        except:
            start_r = 0
    # start_r = 0
    for r in range(start_r, args.repeat):

        print('************************************')
        print('* STARTING TRIAL ' + str(r+1))
        print('************************************')

        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # set up a trainer
        trainer = Trainer(args, seed, metric_keys, save_keys)

        # init total run metrics storage
        max_task = trainer.max_task
        if r == 0: 
            for mkey in metric_keys: 
                avg_metrics[mkey]['global'] = np.zeros((max_task,args.repeat))#初始化0值用np.zeros()
                if (not (mkey in global_only)):
                    avg_metrics[mkey]['pt'] = np.zeros((max_task,max_task,args.repeat))
                    avg_metrics[mkey]['pt-local'] = np.zeros((max_task,max_task,args.repeat))

        # train model
        avg_metrics = trainer.train(avg_metrics)  

        # evaluate model
        avg_metrics = trainer.evaluate(avg_metrics)    

        # save results
        for mkey in metric_keys: 
            m_dir = args.log_dir+'/results-'+mkey+'/'
            if not os.path.exists(m_dir): os.makedirs(m_dir)
            for skey in save_keys:
                if (not (mkey in global_only)) or (skey == 'global'):
                    save_file = m_dir+skey+'.yaml'
                    result=avg_metrics[mkey][skey]
                    yaml_results = {}
                    if len(result.shape) > 2:
                        yaml_results['mean'] = result[:,:,:r+1].mean(axis=2).tolist()
                        if r>1: yaml_results['std'] = result[:,:,:r+1].std(axis=2).tolist()
                        yaml_results['history'] = result[:,:,:r+1].tolist()
                    else:
                        yaml_results['mean'] = result[:,:r+1].mean(axis=1).tolist()
                        if r>1: yaml_results['std'] = result[:,:r+1].std(axis=1).tolist()
                        yaml_results['history'] = result[:,:r+1].tolist()
                    with open(save_file, 'w') as yaml_file:
                        yaml.dump(yaml_results, yaml_file, default_flow_style=False)

        # Print the summary so far
        print('===Summary of experiment repeats:',r+1,'/',args.repeat,'===')
        for mkey in metric_keys: 
            print(mkey, ' | mean:', avg_metrics[mkey]['global'][-1,:r+1].mean(), 'std:', avg_metrics[mkey]['global'][-1,:r+1].std())
    
    


