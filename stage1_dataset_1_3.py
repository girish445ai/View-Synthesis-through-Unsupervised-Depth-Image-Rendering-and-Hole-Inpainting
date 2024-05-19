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
from torch.nn.functional import pad
import torch
from torchvision.utils import save_image
from PIL import Image

class KittiLoader(Dataset):
    def __init__(self, data_root_dir,depth_root_dir,mode, trans_range={"x":-1, "y":-1, "z":-1, "a":-1, "b":-1, "c":-1}, width = 2048, height = 1024, device = "cuda",transform=None):
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
        self.left_depth_paths = []

        depth_left_dir = depth_data_list[1] 
        
        self.left_depth_paths.append(sorted([os.path.join(depth_root_dir,mode,depth_left_dir, fname) for fname\
                        in os.listdir(os.path.join(depth_root_dir,mode,depth_left_dir))]))
        self.left_depth_paths_ = self.left_depth_paths[0]   
        if mode == 'train' or 'val':

            self.right_paths = []
            data_right_dir = data_list[0]
            self.right_paths.append(sorted([os.path.join(data_root_dir,mode,data_right_dir, fname) for fname\
                            in os.listdir(os.path.join(data_root_dir,mode,data_right_dir))]))
            self.right_paths_ = self.right_paths[0]    
            self.right_depth_paths = []
            depth_right_dir = depth_data_list[0]
            self.right_depth_paths.append(sorted([os.path.join(depth_root_dir,mode,depth_right_dir, fname) for fname\
                        in os.listdir(os.path.join(depth_root_dir,mode,depth_right_dir))]))
            self.right_depth_paths_ = self.right_depth_paths[0]
            assert len(self.right_paths_) == len(self.left_paths_)
            assert len(self.right_depth_paths_) == len(self.left_depth_paths_)

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
        fname_left_depth = self.left_depth_paths_[idx]
        left_image = image_to_tensor(self.left_paths_[idx], unsqueeze=False)  # [3,h,w]
        left_depth = disparity_to_tensor(self.left_depth_paths_[idx], unsqueeze=False)  # [1,h,w]
        # print(left_depth.shape)
        # quit()
        right_image = image_to_tensor(self.right_paths_[idx],unsqueeze=False)
        right_depth = disparity_to_tensor(self.right_depth_paths_[idx], unsqueeze=False)  # [1,h,w]
        left_image, left_depth = self.preprocess_rgbd(left_image, left_depth)
        right_image, right_depth = self.preprocess_rgbd(right_image,right_depth)

        if self.mode == 'train' or 'val':
            return left_image, left_depth, right_image, right_depth, fname, fname_left_depth
        else:
            return left_image, left_depth, fname, fname_left_depth
    
    
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
    

    def depth_to_disparity(self,depth_map, baseline = -0.24, focal_length=4730.69819742853):
        depth_image_list = [depth_map[i, 0] for i in range(depth_map.shape[0])]
        disp_image_list = [baseline * focal_length / d for d in depth_image_list]
        pil_image_list = [transforms.ToPILImage()(disp_map) for disp_map in disp_image_list]
        disp = torch.stack([torch.tensor(np.array(pil_image), dtype=torch.float32) for pil_image in pil_image_list])
        return disp
    
    def Wrap_image(self,input_images, x_offset, wrap_mode='edge', tensor_type = 'torch.cuda.FloatTensor'):
        num_batch, num_channels, height, width = input_images.size()

        # Handle both texture border types
        edge_size = 0
        if wrap_mode == 'border':
            edge_size = 1
            # Pad last and second-to-last dimensions by 1 from both sides
            input_images = pad(input_images, (1, 1, 1, 1))
        elif wrap_mode == 'edge':
            edge_size = 0
        else:
            return None
        # print(input_images.shape)

        # Put channels to slowest dimension and flatten batch with respect to others
        input_images = input_images.permute(1, 0, 2, 3).contiguous()
        im_flat = input_images.view(num_channels, -1)
        # print(input_images.shape)
        # quit()
        # Create meshgrid for pixel indicies (PyTorch doesn't have dedicated
        # meshgrid function)
        x = torch.linspace(0, width - 1, width).repeat(height, 1).type(tensor_type)
        y = torch.linspace(0, height - 1, height).repeat(width, 1).transpose(0, 1).type(tensor_type)
        # Take padding into account
        x = x + edge_size
        y = y + edge_size
        # Flatten and repeat for each image in the batch
        x = x.view(-1).repeat(1, num_batch)
        y = y.contiguous().view(-1).repeat(1, num_batch)
        
        # Now we want to sample pixels with indicies shifted by disparity in X direction
        # For that we convert disparity from % to pixels and add to X indicies
        x = x + x_offset.contiguous().view(-1)
        # Make sure we don't go outside of image
        x = torch.clamp(x, 0.0, width - 1 + 2 * edge_size)
        # Round disparity to sample from integer-valued pixel grid
        y0 = torch.floor(y)
        # In X direction round both down and up to apply linear interpolation
        # between them later
        x0 = torch.floor(x)
        x1 = x0 + 1
        # After rounding up we might go outside the image boundaries again
        x1 = x1.clamp(max=(width - 1 + 2 * edge_size))

        # Calculate indices to draw from flattened version of image batch
        dim2 = (width + 2 * edge_size)
        dim1 = (width + 2 * edge_size) * (height + 2 * edge_size)
        # Set offsets for each image in the batch
        base = dim1 * torch.arange(num_batch).type(tensor_type)
        base = base.view(-1, 1).repeat(1, height * width).view(-1)
        # One pixel shift in Y  direction equals dim2 shift in flattened array
        base_y0 = base + y0 * dim2
        # Add two versions of shifts in X direction separately
        idx_l = base_y0 + x0
        idx_r = base_y0 + x1

        # Sample pixels from images
        pix_l = im_flat.gather(1, idx_l.repeat(num_channels, 1).long())
        pix_r = im_flat.gather(1, idx_r.repeat(num_channels, 1).long())

        # Apply linear interpolation to account for fractional offsets
        weight_l = x1 - x
        weight_r = x - x0
        output = weight_l * pix_l + weight_r * pix_r

        # Reshape back into image batch and permute back to (N,C,H,W) shape
        output = output.view(num_channels, num_batch, height, width).permute(1,0,2,3)

        return output
    
    
    def collect_data(self, batch):
        batch = default_collate(batch)
        if self.mode == "train" or "val":
            left_image, left_depth, right_image, right_depth, fname, fname_left_depth = batch
            left_image = left_image.to(self.device)
            left_depth = left_depth.to(self.device)
            right_image = right_image.to(self.device)
            right_depth = right_depth.to(self.device)
            
            left_disp = self.depth_to_disparity(left_depth)
            right_disp = self.depth_to_disparity(right_depth)

            left_disp = left_disp.to(self.device)
            right_disp = right_disp.to(self.device)
            rgbd_left = torch.cat([left_image, left_depth], dim=1)  # [b,4,h,w]
            rgbd_right = torch.cat([right_image, right_depth], dim=1)  # [b,4,h,w]
        else:
            left_image, left_depth, fname, fname_left_depth = batch
            left_image = left_image.to(self.device)
            left_depth = left_depth.to(self.device)
            left_disp = self.depth_to_disparity(left_depth)
            right_disp = self.depth_to_disparity(right_depth)
            left_disp = left_disp.to(self.device)
            right_disp = right_disp.to(self.device)
            rgbd_left = torch.cat([left_image, left_depth], dim=1)  # [b,4,h,w]

        warp_image = self.Wrap_image(left_image,right_disp)
        warp_mask = torch.where(warp_image == 0, warp_image, torch.tensor(255.0).to(self.device))
        # warp_mask = warp_mask.mean(dim=1, keepdim=True).float()

        # print(warp_image.shape, warp_mask.shape)
        # quit()
        warp_mask = warp_mask.to(self.device)
        warp_image = warp_image.to(self.device)

        if self.mode == "train" or "val":
            return {
                "rgb_left":left_image,
                # "depth_left":left_depth,
                'disp_left':left_disp,
                "rgb_right":right_image,
                # "depth_right": right_depth,
                'disp_right':right_disp,
                "mask": warp_mask,
                "warp_rgb": warp_image,
                "fname": fname
            }
        else:
            return {
                "rgb_left": left_image,
                # "depth_left": left_depth,
                'disp_left':left_disp,
                "mask": warp_mask,
                "warp_rgb": warp_image,
                "fname": fname
            }










