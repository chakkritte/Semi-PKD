import time
import sys
import logging
import os
import torch
import torch.nn as nn
from argparse import ArgumentParser
import PKD.datasets.data as data
from PKD.core.eval import validate
from PKD.core.trainer import train_labeled, train_unlabeled
from PKD.models.build_model import build_model, cal_flops_params
from PKD.utils.utils import OwnLogging
import PKD.utils.utils as utils
from PKD.utils.carbon import CarbonAI
import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dataset_dir", type=str, default="/home/kongpasom/PKD/data/")
parser.add_argument('--input_size_h',default=384, type=int)
parser.add_argument('--input_size_w',default=384, type=int)
parser.add_argument('--no_workers',default=12, type=int)
parser.add_argument('--no_epochs',default=20, type=int)
parser.add_argument('--log_interval',default=20, type=int)
parser.add_argument('--lr_sched',default=True, type=bool)
parser.add_argument('--model_val_path',default="model.pt", type=str)
parser.add_argument('--model_salicon_path',default="model_efb4.pt", type=str)
parser.add_argument('--output_dir', type=str, default="outputs")

parser.add_argument('--kldiv', action='store_true', default=True)
parser.add_argument('--cc', action='store_true', default=True)
parser.add_argument('--nss', action='store_true', default=True)
parser.add_argument('--sim', action='store_true', default=False)
parser.add_argument('--l1', action='store_true', default=False)
parser.add_argument('--auc', action='store_true', default=False)

parser.add_argument('--dataset',default="salicon", type=str)
parser.add_argument('--teacher',default="efb4", type=str)
parser.add_argument('--student',default="eeeac2", type=str)
parser.add_argument('--readout',default="simple", type=str)
parser.add_argument('--output_size', default=(480, 640))

parser.add_argument('--amp',default=True, type=bool)
parser.add_argument('--seed',default=3407, type=int)

args = parser.parse_args()

torch.multiprocessing.freeze_support()

# utils.fix_seed(args.seed)

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

start = time.time()

codecarbon = CarbonAI(country_iso_code="THA", args=args)
codecarbon.start()

def model_load_state_dict(student, path_state_dict):
    student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
    logging.info("loaded pre-trained")

if args.dataset != "salicon":
    args.output_size = (384, 384)

student = build_model(args.student, args)

train_loader, val_loader, output_size = data.create_dataset(args)

if args.dataset != "salicon":
    args.output_size = (384, 384)

model_load_state_dict(student, args.model_salicon_path)

cal_flops_params(student, args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    student = nn.DataParallel(student)

student.to(device)

params_group = [
    {"params": list(filter(lambda p: p.requires_grad, student.parameters())), "lr" : args.learning_rate },
]

optimizer = torch.optim.AdamW(params_group)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.no_epochs))

logging.info(device)

with torch.no_grad():
    cc_loss = validate(student, optimizer, val_loader, 0, device, None, 0)