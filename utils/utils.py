import matplotlib.pyplot as plt
import numpy as np
import random
import os
import torch

from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR, \
    ExponentialLR, CosineAnnealingWarmRestarts
from warmup_scheduler import GradualWarmupScheduler

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)

def save_checkpoint(epoch_num, model, optimizer, work_dir):
    checkpointName = 'ep{}.pth.tar'.format(epoch_num)
    checkpointpath = f'{work_dir}/checkpoint/'
    if not os.path.exists(checkpointpath):
        os.makedirs(checkpointpath)
    checkpoint = {
        'epoch': epoch_num,
        'model': model.state_dict(),
        'lr': optimizer.param_groups[0]['lr']
    }
    torch.save(checkpoint, os.path.join(checkpointpath, checkpointName))


def loadCheckpoint(model, optimizer, work_dir="", epoch=-1, checkpointPath=""):
    reload = False

    if not checkpointPath:
        assert work_dir
        model_dir_name = f'{work_dir}/checkpoint/'
        if not os.path.exists(model_dir_name):
            os.mkdir(model_dir_name)

        model_dir = os.listdir(model_dir_name)  # 列出文件夹下文件名
        if len(model_dir) == 0:
            return 0, model, optimizer
        model_dir.sort(key=lambda x: int(x[2:-8]))  # 文件名按数字排序
        if epoch == -1:
            checkpointName = model_dir[-1]  # 获取文件 , -1 获取最后一个文件
        else:
            checkpointName = 'ep{}.pth.tar'.format(epoch)  # 获取文件 , -1 获取最后一个文件
        checkpointPath = os.path.join(model_dir_name, checkpointName)
        reload = True

    if os.path.isfile(checkpointPath):
        print(f"--------------- Loading {checkpointPath}...")
        checkpoint = torch.load(checkpointPath, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.param_groups[0]['lr'] = checkpoint['lr']
        print('---------------- Checkpoint loaded')
    else:
        raise OSError('Checkpoint not found')

    if reload:
        epoch = checkpoint['epoch']
    else:
        epoch = 0
    return epoch, model, optimizer

def build_scheduler(optimizer, lr_scheduler):
    name_scheduler = lr_scheduler.type
    scheduler = None

    if name_scheduler == 'StepLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = StepLR(optimizer=optimizer, step_size=lr_scheduler.step_size, gamma=lr_scheduler.gamma)
    elif name_scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=lr_scheduler.T_max)
    elif name_scheduler == 'ReduceLROnPlateau':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step(val_loss)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode=lr_scheduler.mode)
    elif name_scheduler == 'LambdaLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_scheduler.lr_lambda)
    elif name_scheduler == 'MultiStepLR':
        # >>> train(...)
        # >>> validate(...)
        # >>> scheduler.step()
        scheduler = MultiStepLR(optimizer=optimizer, milestones=lr_scheduler.milestones, gamma=lr_scheduler.gamma)
    elif name_scheduler == 'CyclicLR':
        # >>> for epoch in range(10):
        # >>>   for batch in data_loader:
        # >>>       train_batch(...)
        # >>>       scheduler.step()
        scheduler = CyclicLR(optimizer=optimizer, base_lr=lr_scheduler.base_lr, max_lr=lr_scheduler.max_lr)
    elif name_scheduler == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer=optimizer, gamma=lr_scheduler.gamma)
    elif name_scheduler == 'CosineAnnealingWarmRestarts':
        # >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
        # >>> for epoch in range(20):
        #     >>> scheduler.step()
        # >>> scheduler.step(26)
        # >>> scheduler.step()  # scheduler.step(27), instead of scheduler(20)
        scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=lr_scheduler.T_0,
                                                T_mult=lr_scheduler.T_mult)

    if lr_scheduler.warmup_epochs != 0:
        scheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=1, total_epoch=lr_scheduler.warmup_epochs,
                                           after_scheduler=scheduler)

    if scheduler is None:
        raise Exception('scheduler is wrong')
    return scheduler

def normalize_tensor(tensor, rescale=False, zero_fill=False):
    tmin = torch.min(tensor)
    if rescale or tmin < 0:
        tensor -= tmin

    if zero_fill:
        tensor = torch.where(tensor == 0, tensor.max() * 1e-4, tensor)
    tsum = tensor.sum()
    if tsum > 0:
        return tensor / tsum
    print("Zero tensor")
    tensor.fill_(1. / tensor.numel())
    return tensor
