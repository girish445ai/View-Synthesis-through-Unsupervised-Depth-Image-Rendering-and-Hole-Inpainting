import sys
import logging
import pathlib
import random
import shutil
import time
import numpy as np
import argparse

import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F

import torchvision
from torchvision import transforms

from torch import nn
from torch.autograd import Variable
from torch import optim
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from stage1_dataset_1_3 import KittiLoader
import os
from os.path import join

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR, StepLR, CosineAnnealingLR
#from losses import PerceptualLoss, FocalFrequencyLoss, SSIM
# from test_network import PConvUNet, VGG16FeatureExtractor
from networks import PConvUNet, VGG16FeatureExtractor
from torchvision.utils import save_image
from math import sqrt
from loss import InpaintingLoss
import opt
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_datasets(args):

    batch_Size = args.batch_size
    train_data = KittiLoader(
        data_root_dir=args.data_root_dir,
        depth_root_dir= args.depth_root_dir,
        mode = "train"
    )
    val_data = KittiLoader(
        data_root_dir=args.data_root_dir,
        depth_root_dir= args.depth_root_dir,
        mode = "val"
    )

    return train_data ,val_data


def create_data_loaders(args):
    
    train_data,val_data = create_datasets(args)
    # display_data = [val_data[i] for i in range(0, len(val_data), len(val_data) // 4)]
    # display_data = [val_data[i] for i in range(len(val_data)//2, len(val_data), 2)]
    
    train_loader = DataLoader(dataset=train_data,
                              batch_size=int(args.batch_size),
                              #num_workers=8,
                              shuffle=True,
                              collate_fn=train_data.collect_data
                            )
     
    val_loader = DataLoader(dataset=val_data, 
                            batch_size=int(args.batch_size), 
                            #num_workers=8, 
                            shuffle=False,
                            collate_fn=val_data.collect_data
                            )
    display_loader = DataLoader(dataset=val_data, 
                            batch_size=1, 
                            #num_workers=8, 
                            shuffle=True,
                            collate_fn=val_data.collect_data

                            )
    return train_loader, val_loader, display_loader

def train_epoch(args, epoch, model, data_loader, optimizer, writer,loss_criterion):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    
    for iter, data in enumerate(tqdm(data_loader)):

        left_image, left_disp,right_image,right_disp,mask = data["rgb_left"], data["disp_left"],data["rgb_right"], data["disp_right"],data["mask"]
        w_image = data["warp_rgb"]
        # print(w_image.shape)
        # save_image(w_image, '/data2/pconv_files_new/Test_Warped_Image.png')
        # save_image(right_disp, '/data2/pconv_files_new/Test_Right_Disp.png')
        # save_image(left_disp, '/data2/pconv_files_new/Test_Left_Disp.png')
        # save_image(left_image, '/data2/pconv_files_new/Test_Left_Image.png')
        # save_image(mask, '/data2/pconv_files_new/Test_Mask.png')

        # quit()
        fname = data["fname"]
        # print(fname)
        # quit()
        output,_ = model(w_image,mask)
        inpainted_image = w_image * (mask) + output * (1-mask)
        # print(inpainted_image.shape)
        max_val = torch.max(w_image)
        min_val = torch.min(w_image)
        # print(f"max:{max_val},min:{min_val}")
        # quit()
        loss_dict = loss_criterion(w_image,mask,output,right_image)

        loss = 0.0
        for key, coef in opt.LAMBDA_DICT.items():
            value = coef * loss_dict[key]
            loss += value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('TrainLoss',loss.item(),global_step + iter )


        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'TrainLoss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
        #break

    return loss, time.perf_counter() - start_epoch                 #, acc




def evaluate(args, epoch, model, data_loader, writer,loss_criterion):

    model.eval()
    losses = []
    img_losses = []
    disp_losses = []
    start_iter = time.perf_counter()
    psnr_score = []
    with torch.no_grad():

        for iter, data in enumerate(tqdm(data_loader)):
            left_image, left_disp,right_image,right_disp,mask = data["rgb_left"], data["disp_left"],data["rgb_right"], data["disp_right"],data["mask"]
            w_image = data["warp_rgb"]
            
            output,_ = model(w_image,mask)
            inpainted_image = w_image * (mask) + output * (1-mask)
            
            loss_dict = loss_criterion(w_image,mask,output,right_image)

            loss = 0.0
            for key, coef in opt.LAMBDA_DICT.items():
                value = coef * loss_dict[key]
                loss += value

                losses.append(loss.item())

        writer.add_scalar('validation_Loss',np.mean(losses),epoch)
    
        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'ValidationLoss = {loss.item():.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',

            )
        
    return np.mean(losses), time.perf_counter() - start_iter,inpainted_image,w_image

def visualize(args, epoch, model, data_loader, writer):
    
    def save_image(image, tag):
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            
            left_image, left_disp,right_image,right_disp,mask = data["rgb_left"], data["disp_left"],data["rgb_right"], data["disp_right"],data["mask"]
            w_image = data["warp_rgb"]

            output,_ = model(w_image,mask)
            # inpainted_image = w_image * (mask) + output * (1-mask)
            inpainted_image = w_image * (mask) + output * (1-mask)

            w_image = (w_image - w_image.min()) / (w_image.max() - w_image.min())
            # output = (output - output.min()) / (output.max() - output.min())


            save_image(left_image, 'Left_Image')
            save_image(w_image, 'Warped_Image')
            save_image(right_image, 'Right_Image')
            save_image(inpainted_image,"Inpainted_Right_Image")
            save_image(output,"Inpaint_Gen_output")
            save_image(mask,"Mask")
            # save_image(right_image*mask,"test")
            # save_image(torch.abs(right_image.float() - inpainted_image.float()), 'Error')
            break



def save_model(args, exp_dir, epoch, model, optimizer,best_validation_loss,is_new_best,inpainted_image,w_image):

    out = torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_validation_loss': best_validation_loss,
            'exp_dir':exp_dir,
            'inpainted_image':inpainted_image,
            'w_image':w_image
        },
        f=exp_dir / 'model.pt'
    )

    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')
        
        
def build_model(args):

    # Create the models
    # model = InpaintGenerator().to(args.device)
    model = PConvUNet().to(args.device)
    loss_criterion = InpaintingLoss(VGG16FeatureExtractor()).to(args.device)
    
    return model, loss_criterion

def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model, loss_criterion  = build_model(args)

    model.load_state_dict(checkpoint['model'])

    optimizer = build_optim(args, model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])

    return checkpoint, model, optimizer 

def build_optim(args, params):
    
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, params), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(params, args.lr, rho=0.9, eps=1e-06, weight_decay=args.weight_decay, foreach=None,maximize=False)
    elif args.optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(params, lr=args.lr, lr_decay=0, weight_decay=args.weight_decay, initial_accumulator_value=0, eps=1e-10, foreach=None)

    return optimizer

def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(args.exp_dir / 'summary'))

    if args.resume:
        print('resuming model, batch_size', args.batch_size)
        checkpoint, model, optimizer = load_model(args.checkpoint)
        args = checkpoint['args']
        args.batch_size = 1
        best_validation_loss= checkpoint['best_validation_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model, loss_criterion= build_model(args)   
        optimizer = build_optim(args, model.parameters())
        best_validation_loss = 1e9 #Inital validation loss
        start_epoch = 0

    logging.info(args)
    logging.info(model)

    train_loader, validation_loader,display_loader = create_data_loaders(args)

    if args.lr_sched == "noS":
        scheduler = None
    elif args.lr_sched == "expS":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.lr_sched == "cosS":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    elif args.lr_sched == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    
    for epoch in range(start_epoch, args.num_epochs):

        train_loss,train_time = train_epoch(args, epoch, model, train_loader,optimizer,writer,loss_criterion)
        validation_loss,validation_time,inpainted_image,w_image = evaluate(args, epoch, model, validation_loader, writer,loss_criterion)
        if args.lr_sched != "noS":
            scheduler.step

        visualize(args, epoch, model, display_loader, writer)

        is_new_best = validation_loss < best_validation_loss
        best_validation_loss = min(best_validation_loss,validation_loss)
        save_model(args, args.exp_dir, epoch, model, optimizer,best_validation_loss,is_new_best,inpainted_image,w_image)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] '
            f'TrainLoss = {train_loss:.4g} ' 
            f'TrainTime = {train_time:.4f}s '
            f'validation_loss= {validation_loss:.4g} '
            f'validationTime = {validation_time:.4f}s ',
        )
    writer.close()
    
def create_arg_parser():

    parser = argparse.ArgumentParser(description='Train setup for QATool')
    parser.add_argument('--seed',default=42,type=int,help='Seed for random number generators')
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument('--num-epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--weight-decay', type=float, default=1e-3,
                        help='Strength of weight decay regularization')
    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--data-root-dir',type=str,help='Path to data directory')
    parser.add_argument('--depth-root-dir',type=str,help='Path to depth data directory')
    parser.add_argument('--preds-output-dir',type=str,help='Path to directory where predicted node classes are to be saved')
    parser.add_argument("--lr_sched", default="expS", type=str)
    parser.add_argument("--optimizer", default="Adam", type=str)
    # parser.add_argument("--dropout", default=True, type=lambda x: bool(str2bool(x)))    
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
#     print (args)
    main(args)