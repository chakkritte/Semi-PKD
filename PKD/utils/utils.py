import torch
import random
import os
import numpy as np
import shutil

class AverageMeter(object):

    '''Computers and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter2(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def fix_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    else: 
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

class OwnLogging(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.results = []

    def update(self, epoch, lr, cc, kd, nss, sim, auc, total):
        self.results.append([epoch, lr, cc, kd, nss, sim, auc, total])
        
    def done(self, name):
        np.savetxt(name, np.array(self.results), delimiter=',', header="epoch,lr,cc,kd,nss,sim,auc,total", comments="", fmt='%s')


def save_state_dict(student, args):
    if torch.cuda.device_count() > 1: 
        params = {
            'student': student.module.state_dict()
        }
    else :
        params = {
            'student': student.state_dict(),
        }
    torch.save(params, os.path.join(args.save, args.model_val_path))

def save_state_dict_pkd(student , teacher, args):
    if torch.cuda.device_count() > 1: 
        params = {
            'student': student.module.state_dict(),
            'teacher': teacher.module.state_dict()
        }
    else :
        params = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict()
        }
    torch.save(params, os.path.join(args.save, args.model_val_path))