import time
import sys
import logging
import os
import torch
import torch.nn as nn
from argparse import ArgumentParser
import PKD.datasets.data as data
from PKD.core.eval import validate
from PKD.core.trainer import train_labeled_selfkd
from PKD.models.build_model import build_model, cal_flops_params
from PKD.utils.utils import OwnLogging
import PKD.utils.utils as utils
from PKD.utils.carbon import CarbonAI
from PKD.utils.swa_utils import AveragedModel
# Import the configuration
import config

import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()

parser.add_argument("--dataset_dir", type=str, default=config.DATASET_PATH)
parser.add_argument('--no_workers',type=int, default=config.NO_WORKER)
parser.add_argument('--log_interval', type=int, default=config.LOS_INTERVAL)

parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument('--input_size_h',default=384, type=int)
parser.add_argument('--input_size_w',default=384, type=int)
parser.add_argument('--no_epochs',default=20, type=int)
parser.add_argument('--lr_sched',default=True, type=bool)
parser.add_argument('--model_val_path',default="model.pt", type=str)
parser.add_argument('--model_salicon_path',default="model_salicon.pt", type=str)
parser.add_argument('--output_dir', type=str, default="outputs")

parser.add_argument('--kldiv', action='store_true', default=True)
parser.add_argument('--cc', action='store_true', default=True)
parser.add_argument('--nss', action='store_true', default=True)
parser.add_argument('--sim', action='store_true', default=False)
parser.add_argument('--l1', action='store_true', default=False)
parser.add_argument('--auc', action='store_true', default=False)

parser.add_argument('--dataset',default="salicon", type=str)
parser.add_argument('--student',default="efb4", type=str)
parser.add_argument('--readout',default="simple", type=str)
parser.add_argument('--output_size', default=(480, 640))

parser.add_argument('--amp',default=True, type=bool)
parser.add_argument('--seed',default=3407, type=int)

parser.add_argument('--self_kd',default=True, type=bool)
parser.add_argument('--ema_decay',default=0.999, type=float)

args = parser.parse_args()

torch.multiprocessing.freeze_support()

#utils.fix_seed(args.seed)

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

csv_log = OwnLogging()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

args.save = '{}/{}-{}-teacher-swa-{}'.format(args.output_dir, args.dataset, args.student, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.getLogger('PIL').setLevel(logging.WARNING)

logging.info("Hyperparameter: %s" % args)
logging.info('-'*80)

start = time.time()

codecarbon = CarbonAI(country_iso_code="THA", args=args)
codecarbon.start()

def model_load_state_dict(student, path_state_dict):
    student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
    logging.info("loaded pre-trained student")

if args.dataset != "salicon":
    args.output_size = (384, 384)

student = build_model(args.student, args)

logging.info(args.dataset)

train_loader, val_loader, output_size = data.create_dataset(args)

if args.dataset != "salicon":
    args.output_size = (384, 384)

if args.dataset != "salicon":
    model_load_state_dict(student, args.model_salicon_path)

cal_flops_params(student, args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    student = nn.DataParallel(student)

student.to(device)

if args.self_kd:
    swa_model = AveragedModel(model=student, use_buffers=True, device=device)
else:
    swa_model = None

params_group = [
    {"params": list(filter(lambda p: p.requires_grad, student.parameters())), "lr" : args.learning_rate },
]

optimizer = torch.optim.AdamW(params_group)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.no_epochs))

logging.info(device)

for epoch in range(0, args.no_epochs):
    lr = optimizer.param_groups[0]['lr']
    logging.info(f"Epoch {epoch}: Learning rate : {lr}")

    loss = train_labeled_selfkd(student, optimizer, train_loader, epoch, device, args, scaler, swa_model)

    if args.lr_sched:
        scheduler.step()

    with torch.no_grad():
        if args.self_kd:
            _ = validate(swa_model, optimizer, val_loader, epoch, device, None, lr)
        cc_loss = validate(student, optimizer, val_loader, epoch, device, csv_log, lr)

        if epoch == 0:
            best_loss = cc_loss
        if best_loss >= cc_loss:
            best_loss = cc_loss
            
            if args.self_kd:
                swa_model.update_parameters(student)

            logging.info('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
            utils.save_state_dict(student, args)
        print()

csv_log.done(os.path.join(args.save, 'val.csv'))

end = time.time()
logging.info("Time = %.2f Sec", (end - start))

codecarbon.stop()
logging.info("Emission = %.4f kgCO2" % codecarbon.emissions)
logging.info("Power = %.2f W" % codecarbon.power)
