import torch
import time
import sys
import logging
from tqdm import tqdm
from PKD.losses.loss import loss_func, loss_func_self

def train_labeled(student, optimizer, loader, epoch, device, args, scaler, ema_model):
    student.train()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt, fixations) in enumerate(tqdm(loader)):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.amp):
            if args.ema_kd and epoch > 0:
                with torch.no_grad():
                    pred_map_ema = ema_model(img)
                    soft_logits_ema = pred_map_ema.clone().detach()
                    gt = soft_logits_ema + (gt - soft_logits_ema) * 0.5

            pred_map_student = student(img)
            loss = loss_func(pred_map_student, gt, fixations, args)

        scaler.scale(loss).backward()
        total_loss += loss.item()
        cur_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()

        if args.ema_kd and epoch > 0:
            ema_model.update_parameters(student)

        if idx%args.log_interval==(args.log_interval-1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60, round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)


def train_unlabeled(student, optimizer, loader, epoch, device, args, scaler, ema_model, teacher):
    student.train()
    teacher.eval()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, _) in enumerate(tqdm(loader)):
        img = img.to(device)
        
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.amp):
            with torch.no_grad():
                pseudo_gt = teacher(img).clone().detach()

                if args.ema_kd and epoch > 0:
                    pred_map_ema = ema_model(img).clone().detach()
                    pseudo_gt = pred_map_ema + (pseudo_gt - pred_map_ema) * 0.5
                    
            pred_map_student = student(img)
            loss = loss_func_self(pred_map_student, pseudo_gt, None, args)

        scaler.scale(loss).backward()
        total_loss += loss.item()
        cur_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()

        if args.ema_kd and epoch > 0:
            ema_model.update_parameters(student)

        if idx%args.log_interval==(args.log_interval-1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60, round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)


def train_labeled_selfkd(student, optimizer, loader, epoch, device, args, scaler, swa_model):
    student.train()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt, fixations) in enumerate(tqdm(loader)):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.amp):
            if args.self_kd and epoch > 0:
                with torch.no_grad():
                    pred_map_swa = swa_model(img)
                    soft_logits_swa = pred_map_swa.clone().detach()
                    gt = soft_logits_swa + (gt - soft_logits_swa) * 0.5
                    
            pred_map_student = student(img)
            loss = loss_func(pred_map_student, gt, fixations, args)

        scaler.scale(loss).backward()
        total_loss += loss.item()
        cur_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()

        if idx%args.log_interval==(args.log_interval-1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60, round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)