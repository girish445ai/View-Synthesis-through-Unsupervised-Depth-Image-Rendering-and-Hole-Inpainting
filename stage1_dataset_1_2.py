import sys
sys.path.append(".")
sys.path.append("..")
import os
import glob
import math
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader, default_collate
from torchvision.utils import save_image
from skimage.feature import canny
import numpy as np
from torchvision import transforms
from PIL import Image
import opt
import Warper
from utils import (
    RGBDRenderer, 
    image_to_tensor, 
    disparity_to_tensor,
    transformation_from_parameters,
)
import os
from PIL import Image
from glob import glob

class KittiLoader(Dataset):
    def __init__(self, data_root_dir,depth_root_dir,mode, trans_range={"x":-1, "y":-1, "z":-1, "a":-1, "b":-1, "c":-1}, width = 512, height =512, device = "cuda",transform=None):
        self.width = width
        self.height = height
        self.data_root_dir = data_root_dir
        self.depth_root_dir = depth_root_dir
        self.mode = mode
        self.device = device
        self.trans_range = trans_range
        self.renderer = RGBDRenderer(device)
        data_list = os.listdir(os.path.join(data_root_dir,mode))
        
        depth_data_list = os.listdir(os.path.join(depth_root_dir,mode))

        
        self.left_paths = []
        data_left_dir = data_list[1] #source 
        self.left_paths.append(sorted([os.path.join(data_root_dir,mode,data_left_dir, fname) for fname\
                           in os.listdir(os.path.join(data_root_dir,mode,data_left_dir))]))
        self.left_paths_ = self.left_paths[0] 
        self.left_disp_paths = []

        depth_left_dir = depth_data_list[1] 
        
        self.left_disp_paths.append(sorted([os.path.join(depth_root_dir,mode,depth_left_dir, fname) for fname\
                        in os.listdir(os.path.join(depth_root_dir,mode,depth_left_dir))]))
        self.left_disp_paths_ = self.left_disp_paths[0]   
        if mode == 'train' or 'val':

            self.right_paths = []
            data_right_dir = data_list[0]
            self.right_paths.append(sorted([os.path.join(data_root_dir,mode,data_right_dir, fname) for fname\
                            in os.listdir(os.path.join(data_root_dir,mode,data_right_dir))]))
            self.right_paths_ = self.right_paths[0]    
            self.right_disp_paths = []
            depth_right_dir = depth_data_list[0]
            self.right_disp_paths.append(sorted([os.path.join(depth_root_dir,mode,depth_right_dir, fname) for fname\
                        in os.listdir(os.path.join(depth_root_dir,mode,depth_right_dir))]))
            self.right_disp_paths_ = self.right_disp_paths[0]
            assert len(self.right_paths_) == len(self.left_paths_)
            assert len(self.right_disp_paths_) == len(self.left_disp_paths_)

        self.transform = transform
        self.mode = mode

        # self.K = torch.tensor([
        #     [0.58, 0, 0.5],
        #     [0, 0.58, 0.5],
        #     [0, 0, 1]
        # ]).to(device)

    def __len__(self):
        return len(self.left_paths_)

    def __getitem__(self, idx):
    
        fname = self.left_paths_[idx]
        fname_left_disp = self.left_disp_paths_[idx]
        left_image = image_to_tensor(self.left_paths_[idx], unsqueeze=False)  # [3,h,w]
        left_disp = disparity_to_tensor(self.left_disp_paths_[idx], unsqueeze=False)  # [1,h,w]
        right_image = image_to_tensor(self.right_paths_[idx],unsqueeze=False)
        right_disp = disparity_to_tensor(self.right_disp_paths_[idx], unsqueeze=False)  # [1,h,w]
        left_image, left_disp = self.preprocess_rgbd(left_image, left_disp)
        right_image, right_disp = self.preprocess_rgbd(right_image,right_disp)

        if self.mode == 'train' or 'val':
            return left_image, left_disp, right_image, right_disp, fname, fname_left_disp
        else:
            return left_image, left_disp, fname, fname_left_disp
    
    
    def preprocess_rgbd(self, image, disp):
        # NOTE 
        # (1) here we directly resize the image to the target size (self.height, self.width)
        # a better way is to first crop a random patch from the image according to the height-width ratio
        # then resize this patch to the target size
        # (2) another suggestion is, add some code to filter the depth map to reduce artifacts around 
        # depth discontinuities
        image = F.interpolate(image.unsqueeze(0), (self.height, self.width), mode="bilinear").squeeze(0)
        disp = F.interpolate(disp.unsqueeze(0), (self.height, self.width), mode="bilinear").squeeze(0)
        return image, disp
    
    
    def collect_data(self, batch):
        batch = default_collate(batch)
        if self.mode == "train" or "val":
            left_image, left_disp, right_image, right_disp, fname, fname_left_disp = batch
            left_image = left_image.to(self.device)
            left_disp = left_disp.to(self.device)
            right_image = right_image.to(self.device)
            right_disp = right_disp.to(self.device)
            rgbd_left = torch.cat([left_image, left_disp], dim=1)  # [b,4,h,w]
            rgbd_right = torch.cat([right_image, right_disp], dim=1)  # [b,4,h,w]
        else:
            left_image, left_disp, fname, fname_left_disp = batch
            left_image = left_image.to(self.device)
            left_disp = left_disp.to(self.device)
            rgbd_left = torch.cat([left_image, left_disp], dim=1)  # [b,4,h,w]

        warp_image, warp_mask = Warper.demo1_batch(fname, fname_left_disp,self.width, self.height)
        warp_image, warp_mask = torch.from_numpy(warp_image), torch.from_numpy(warp_mask)
        warp_image = warp_image.permute(0,3,1,2)
        warp_mask = warp_mask.unsqueeze(1).byte()
      
        pil_image_list = [transforms.ToPILImage()(warp_mask[i, 0]) for i in range(warp_mask.shape[0])]
        pil_image_list_rgb = [warp_mask.convert("RGB") for warp_mask in pil_image_list]
        warp_mask = torch.stack([transforms.ToTensor()(warp_mask) for warp_mask in pil_image_list_rgb])
      
        # warp_mask = warp_mask.repeat(1,3,1,1)
        warp_mask = warp_mask.to(self.device)
        warp_image = warp_image.to(self.device)

        if self.mode == "train" or "val":
            return {
                "rgb_left":left_image,
                "disp_left":left_disp,
                "rgb_right":right_image,
                "disp_right": right_disp,
                "mask": warp_mask,
                "warp_rgb": warp_image,
                "fname": fname
            }
        else:
            return {
                "rgb_left": left_image,
                "disp_left": left_disp,
                "mask": warp_mask,
                "warp_rgb": warp_image,
                "fname": fname
            }










