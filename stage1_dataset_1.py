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
# from networks import EdgeGenerator
from PIL import Image
import opt

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
    def __init__(self, data_root_dir,depth_root_dir,mode, trans_range={"x":-1, "y":-1, "z":-1, "a":-1, "b":-1, "c":-1}, width = 512, height =1024, device = "cuda",transform=None):
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
        #for data in data_list:
        data_left_dir = data_list[1] #source 
        self.left_paths.append(sorted([os.path.join(data_root_dir,mode,data_left_dir, fname) for fname\
                           in os.listdir(os.path.join(data_root_dir,mode,data_left_dir))]))
        self.left_paths_ = self.left_paths[0]
        # print(self.left_paths)
        # quit()   
        self.left_disp_paths = []
        #for data in depth_data_list:
     
        depth_left_dir = depth_data_list[1] #source 
        
        self.left_disp_paths.append(sorted([os.path.join(depth_root_dir,mode,depth_left_dir, fname) for fname\
                        in os.listdir(os.path.join(depth_root_dir,mode,depth_left_dir))]))
        self.left_disp_paths_ = self.left_disp_paths[0]   
        if mode == 'train' or 'val':

            self.right_paths = []
            #for data in data_list:
            data_right_dir = data_list[0]
            self.right_paths.append(sorted([os.path.join(data_root_dir,mode,data_right_dir, fname) for fname\
                            in os.listdir(os.path.join(data_root_dir,mode,data_right_dir))]))
            self.right_paths_ = self.right_paths[0]    
            self.right_disp_paths = []
            #for data in depth_data_list:
            depth_right_dir = depth_data_list[0]
            self.right_disp_paths.append(sorted([os.path.join(depth_root_dir,mode,depth_right_dir, fname) for fname\
                        in os.listdir(os.path.join(depth_root_dir,mode,depth_right_dir))]))
            self.right_disp_paths_ = self.right_disp_paths[0]
            assert len(self.right_paths_) == len(self.left_paths_)
            assert len(self.right_disp_paths_) == len(self.left_disp_paths_)

        self.transform = transform
        self.mode = mode

        self.K = torch.tensor([
            [0.58, 0, 0.5],
            [0, 0.58, 0.5],
            [0, 0, 1]
        ]).to(device)
        # self.K = torch.tensor([
        #     [4732.08231939179, 0, 2105.636247915520],
        #     [0, 4729.31407546527, 0.5],
        #     [0, 0, 1059.86872977410]
        # ]).to(device)        

    def __len__(self):
        # print(self.left_paths[0])
        # quit()
        return len(self.left_paths_)

    def __getitem__(self, idx):
        # left_image = Image.open(self.left_paths_[idx])
        # print(len(self.left_paths))
        # quit()
        fname = self.left_paths_[idx]
        left_image = image_to_tensor(self.left_paths_[idx], unsqueeze=False)  # [3,h,w]
        left_disp = disparity_to_tensor(self.left_disp_paths_[idx], unsqueeze=False)  # [1,h,w]
        right_image = image_to_tensor(self.right_paths_[idx],unsqueeze=False)
        # print(self.right_disp_paths_[idx].split('/')[-1].split('.')[-1])
        # quit()
        right_disp = disparity_to_tensor(self.right_disp_paths_[idx], unsqueeze=False)  # [1,h,w]
       
        # do some data augmentation, ensure the rgbd spatial resolution is (self.height, self.width)
        # print(left_image.shape)
        # print(left_disp.shape)
        # quit()
        left_image, left_disp = self.preprocess_rgbd(left_image, left_disp)
        right_image, right_disp = self.preprocess_rgbd(right_image,right_disp)
        # print(right_image.shape,right_disp.shape)
        # quit()
        if self.mode == 'train' or 'val':
            return left_image, left_disp, right_image, right_disp, fname
        else:
            return left_image, left_disp, fname
    
    def normalize_rgb_tensor(self,tensor):
        # Ensure the tensor contains floating-point values
        tensor = tensor.float()
        # Compute the maximum and minimum values for each channel
        max_values, _ = torch.max(tensor.view(tensor.size(0), -1), dim=1)
        min_values, _ = torch.min(tensor.view(tensor.size(0), -1), dim=1)

        # Normalize each channel
        normalized_tensor = (tensor - min_values[:, None, None]) / (max_values[:, None, None] - min_values[:, None, None] )


        return normalized_tensor


    def preprocess_rgbd(self, image, disp):
        # NOTE 
        # (1) here we directly resize the image to the target size (self.height, self.width)
        # a better way is to first crop a random patch from the image according to the height-width ratio
        # then resize this patch to the target size
        # (2) another suggestion is, add some code to filter the depth map to reduce artifacts around 
        # depth discontinuities
        image = F.interpolate(image.unsqueeze(0), (self.height, self.width), mode="bilinear").squeeze(0)
        # print(disp.unsqueeze(0).shape)

        disp = F.interpolate(disp.unsqueeze(0), (self.height, self.width), mode="bilinear").squeeze(0)
        # image = self.normalize_rgb_tensor(image)
        # self.transform = transforms.Normalize(mean=opt.MEAN,std=opt.STD)
        # image = self.transform(image)
       
        return image, disp
    
    def get_rand_ext(self, bs):
        # x, y, z = self.trans_range['x'], self.trans_range['y'], self.trans_range['z']
        # a, b, c = self.trans_range['a'], self.trans_range['b'], self.trans_range['c']
        # cix = self.rand_tensor(x, bs)
        # ciy = self.rand_tensor(y, bs)
        # ciz = self.rand_tensor(z, bs)
        # aix = self.rand_tensor(math.pi / a, bs)
        # aiy = self.rand_tensor(math.pi / b, bs)
        # aiz = self.rand_tensor(math.pi / c, bs)
        
        # axisangle = torch.cat([aix, aiy, aiz], dim=-1)  # [b,1,3]
        # translation = torch.cat([cix, ciy, ciz], dim=-1)
        
        # cam_ext = transformation_from_parameters(axisangle, translation)  # [b,4,4]
        cam_ext = torch.tensor([
            [1.0, 0, 0,-0.24],
            [0, 1.0, 0,0.0],
            [0, 0, 1.0,0.0]
        ])
        cam_ext = cam_ext[None, ...].repeat(bs, 1, 1)  # [b,3,3]
        # cam_ext_inv = torch.inverse(cam_ext)  # [b,4,4]
        # return cam_ext[:, :-1]        #, cam_ext_inv[:, :-1]
        return cam_ext
    
    def rand_tensor(self, r, l):
        '''
        return a tensor of size [l], where each element is in range [-r,-r/2] or [r/2,r]
        '''
        if r < 0:  # we can set a negtive value in self.trans_range to avoid random transformation
            return torch.zeros((l, 1, 1))
        rand = torch.rand((l, 1, 1))        
        sign = 2 * (torch.randn_like(rand) > 0).float() - 1
        return sign * (r / 2 + r / 2 * rand)


    def get_edge(self, image_gray, mask):
        image_gray_np = image_gray.squeeze(1).cpu().numpy()  # [b,h,w]
        mask_bool_np = np.array(mask.squeeze(1).cpu(), dtype=np.bool_)  # [b,h,w]
        edges = []
        for i in range(mask.shape[0]):
            cur_edge = canny(image_gray_np[i], sigma=2, mask=mask_bool_np[i])
            edges.append(torch.from_numpy(cur_edge).unsqueeze(0))  # [1,h,w]
        edge = torch.cat(edges, dim=0).unsqueeze(1).float()  # [b,1,h,w]
        return edge.to(self.device)

    
    def inpaint(self, image, mask):
        image_gray = transforms.Grayscale()(image)
        edge = self.get_edge(image_gray, mask)
        
        mask_hole = 1 - mask

        # inpaint edge
        edge_model_input = torch.cat([image_gray, edge, mask_hole], dim=1)  # [b,3,h,w]
        edge_model = EdgeGenerator().to(self.device)
        # edge_model_weight = torch.load(os.path.join("/home/snehasree/sneha_folders/AdaMPI_copy/AdaMPI/warpback/EdgeModel_gen.pth"))
        # edge_model.load_state_dict(edge_model_weight["generator"])
        edge_model.eval()

        edge_inpaint = edge_model(edge_model_input)  # [b,1,h,w]

        # print(edge_inpaint.shape)
        # print(edge.shape)
        # quit()
        # return edge, mask_hole
        return edge_inpaint,mask_hole


    def collect_data(self, batch):
        batch = default_collate(batch)
        if self.mode == "train" or "val":
            left_image, left_disp, right_image, right_disp, fname = batch
            left_image = left_image.to(self.device)
            left_disp = left_disp.to(self.device)
            right_image = right_image.to(self.device)
            right_disp = right_disp.to(self.device)
            rgbd_left = torch.cat([left_image, left_disp], dim=1)  # [b,4,h,w]
            rgbd_right = torch.cat([right_image, right_disp], dim=1)  # [b,4,h,w]
        else:
            left_image, left_disp, fname = batch
            left_image = left_image.to(self.device)
            left_disp = left_disp.to(self.device)
            rgbd_left = torch.cat([left_image, left_disp], dim=1)  # [b,4,h,w]

        b = left_image.shape[0]
        cam_int = self.K.repeat(b, 1, 1)  # [b,3,3]
        # print(left_image.shape)
        # quit()
        # cam_ext, cam_ext_inv = self.get_rand_ext(b)  # [b,3,4]
        cam_ext = self.get_rand_ext(b)  # [b,3,4]

        cam_ext = cam_ext.to(self.device)
        # cam_ext_inv = cam_ext_inv.to(self.device)
        # print(f"int:{cam_int.shape},ext:{cam_ext.shape}")
        # quit()
        # warp to a random novel view
        mesh = self.renderer.construct_mesh(rgbd_left, cam_int)
        warp_image, warp_disp, warp_mask = self.renderer.render_mesh(mesh, cam_int, cam_ext)
        print(left_image.shape, left_disp.shape)
        print(warp_image.shape,warp_mask.shape)
        quit()
        # print(len(image_list))
        # print(warp_image[0].shape)
        # print(image_list[0].shape)
        # quit()
        # self.transform = transforms.Normalize(mean=opt.MEAN,std=opt.STD)
        # image_list = [warp_image[i] for i in range(warp_image.shape[0])]
        # warp_image = torch.stack([self.transform(img) for img in image_list])
        warp_image = torch.clamp(warp_image,min=0,max=1.0)
        pil_image_list = [transforms.ToPILImage()(warp_mask[i, 0]) for i in range(warp_mask.shape[0])]
        pil_image_list_rgb = [warp_mask.convert("RGB") for warp_mask in pil_image_list]
        warp_mask = torch.stack([transforms.ToTensor()(warp_mask) for warp_mask in pil_image_list_rgb])
        warp_mask = warp_mask.to(self.device)
        # print(warp_mask.shape)
        # quit()
        # edge_inpaint,mask_hole = self.inpaint(warp_image,warp_mask)
       
        # NOTE
        # (1) to train the inpainting network, you only need image, disp, and mask
        # (2) you can add some morphological operation to refine the mask
        if self.mode == "train" or "val":
            return {
                "rgb_left":left_image,
                "disp_left":left_disp,
                "rgb_right":right_image,
                "disp_right": right_disp,
                "mask": warp_mask,
                "warp_rgb": warp_image,
                "warp_disp": warp_disp,
                "fname": fname
                # "edge_inpaint":edge_inpaint,
                # "mask_hole":mask_hole

            }
        else:
            return {
                "rgb_left": left_image,
                "disp_left": left_disp,
                "mask": warp_mask,
                "warp_rgb": warp_image,
                "warp_disp": warp_disp,
                "fname": fname
                # "edge_inpaint":edge_inpaint,
                # "mask_hole":mask_hole
            }










