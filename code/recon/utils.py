import numpy as np, torch.nn, torch.nn.functional, torch.distributions
import random

from torch.utils.data import Dataset, DataLoader
from monai.transforms import Compose, Resize




#########################
#                       #
#      ENV UTILS        #
#                       #
#########################

def get_device():
    '''
    return cuda device if GPU available
    '''
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#########################
#                       #
#    LOGGING UTILS      #
#                       #
#########################

def log_msg(_string):
    '''
    logging function printing date, scriptname & input string to stdout
    '''
    import datetime, os, sys
    print(f'{datetime.date.today().strftime("%a %B %d %H:%M:%S %Z %Y")} {str(os.path.basename(sys.argv[0]))}: {str(_string)}')



#########################
#                       #
#      DATA UTILS       #
#                       #
#########################


def resize(volume, target_size):
    import numpy
    resize_transform = Compose([Resize((target_size[0],
                                        target_size[1],
                                        target_size[2]))])
    if len(volume.shape) == 3:
        volume = numpy.expand_dims(volume, axis=0)
    resized_volume = resize_transform(volume)
    resized_volume = numpy.squeeze(resized_volume)
    return resized_volume


def get_cog(_array):
    '''
    return center of gravity for
    binary lesion mask
    '''
    import numpy
    return(numpy.round(numpy.mean(numpy.where(_array!=0), axis = 1),0))




#########################
#                       #
#    DATASET UTILS      #
#                       #
#########################

# ---- inital data set ---- #
class DeficitDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = self.data[index]
        return img, np.expand_dims(self.labels[index], axis=0)

# ---- lesion pretraining data set ---- #
class LesionDataset(Dataset):
    '''
    Dataset class containing:
        data = lesion masks
    '''
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = self.data[index]
        return img
    @property
    def n_samples(self):
        return(self.data.shape[0])
        

# ---- update more comprehensive dataset ---- #

class DeficitDataset(Dataset):
    '''
    Dataset class containing:
        data = lesion masks
        labels = severity scores
        covar = patient covariates
    
        future updates:
            * including augmentations
            * anatomical information
            * disease information
    '''
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = self.data[index]
        return img, np.expand_dims(self.labels[index], axis=0)

#########################
#                       #
#     MODEL UTILS       #
#                       #
#########################



def count_parameters(model):
    '''
    return number of trainable parameters in given model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def add_coords(x, just_coords=False):
    '''
    2D version: Add coordinate channels to a 4D tensor (B, C, H, W)
    '''
    import torch
    batch_size_shape, channel_in_shape, dim_y, dim_x = x.shape
    # Create coordinate channels
    xx_channel = torch.linspace(0, 1, steps=dim_x, device=x.device).repeat(1, 1, dim_y, 1)
    yy_channel = torch.linspace(0, 1, steps=dim_y, device=x.device).repeat(1, 1, 1, dim_x)
    xx_channel = xx_channel.permute(0, 1, 2, 3)
    yy_channel = yy_channel.permute(0, 1, 2, 3)
    xx_channel = xx_channel.expand(batch_size_shape, 1, dim_y, dim_x)
    yy_channel = yy_channel.transpose(2, 3).expand(batch_size_shape, 1, dim_y, dim_x)
    if just_coords:
        out = torch.cat([xx_channel, yy_channel], dim=1)
    else:
        out = torch.cat([x, xx_channel, yy_channel], dim=1)
    return out


# def add_coords(x, just_coords=False):
#     '''
#     This just the Uber CoordConv method extended to 3D. Definitely use it on the input
#     Using it on other layers of the model can be helpful, but it slows down training
#     :param x:
#     :param just_coords:
#     :return:
#     '''
#     import torch
#     batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = x.shape
#     xx_ones = torch.ones([1, 1, 1, 1, dim_x])
#     yy_ones = torch.ones([1, 1, 1, 1, dim_y])
#     zz_ones = torch.ones([1, 1, 1, 1, dim_z])
#     # ---- intial mapping
#     xy_range = torch.arange(dim_y).float()
#     xy_range = xy_range[None, None, None, :, None]
#     yz_range = torch.arange(dim_z).float()
#     yz_range = yz_range[None, None, None, :, None]
#     zx_range = torch.arange(dim_x).float()
#     zx_range = zx_range[None, None, None, :, None]
#     # ---- x:z mapping
#     xy_channel = torch.matmul(xy_range, xx_ones)
#     xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)
#     xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1, 1)
#     # ---- y:z mapping
#     yz_channel = torch.matmul(yz_range, yy_ones)
#     yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
#     yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)
#     yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1, 1)
#     # ---- z:x mapping
#     zx_channel = torch.matmul(zx_range, zz_ones)
#     zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
#     zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)
#     zz_channel = zz_channel.repeat(batch_size_shape, 1, 1, 1, 1)
#     # ---- i:i mapping
#     xx_channel = xx_channel.to(x.device)
#     yy_channel = yy_channel.to(x.device)
#     zz_channel = zz_channel.to(x.device)
#     xx_channel = xx_channel.float() / (dim_x - 1)
#     yy_channel = yy_channel.float() / (dim_y - 1)
#     zz_channel = zz_channel.float() / (dim_z - 1)
#     if just_coords:
#         out = torch.cat([xx_channel, yy_channel, zz_channel], dim=1)
#     else:
#         out = torch.cat([x, xx_channel, yy_channel, zz_channel], dim=1)
#     return(out)

class SBlock(torch.nn.Module):
    def __init__(self, in_planes, planes, downsample=False, ks=3, stride=1, upsample=False, add_coords=False):
        '''
        This is the Convolutional block that constitutes the meat of the Encoder and Decoder
        :param in_planes:
        :param planes:
        :param downsample:
        :param ks:
        :param stride:
        :param upsample:
        :param add_coords:
        '''
        super(SBlock, self).__init__()
        self.downsample = downsample
        self.upsample = upsample
        if ks == 3:
            pad = 1
        elif ks == 5:
            pad = 2
        else:
            pad = 3
        if add_coords:
            in_planes += 2  # 2D: add 2 channels for coordinates
        self.add_coords = add_coords
        self.c1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_planes, planes, kernel_size=ks, stride=stride, padding=pad),
            torch.nn.BatchNorm2d(planes),
            torch.nn.GELU()
        )
        self.upsample_layer = torch.nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        # if self.add_coords:
        #     x = add_coords(x)
        out = self.c1(x)
        if self.downsample:
            out = torch.nn.functional.avg_pool2d(out, kernel_size=2, stride=2)
        if self.upsample:
            out = self.upsample_layer(out)
        return(out)

# class SBlock(torch.nn.Module):
#     def __init__(self, in_planes, planes, downsample=False, ks=3, stride=1, upsample=False, add_coords=False):
#         '''
#         This is the Convolutional block that constitutes the meat of the Encoder and Decoder
#         :param in_planes:
#         :param planes:
#         :param downsample:
#         :param ks:
#         :param stride:
#         :param upsample:
#         :param add_coords:
#         '''
#         super(SBlock, self).__init__()
#         self.downsample = downsample
#         self.upsample = upsample
#         if ks == 3:
#             pad = 1
#         elif ks == 5:
#             pad = 2
#         else:
#             pad = 3
#         if add_coords:
#             in_planes += 3
#         self.add_coords = add_coords
#         self.c1 = torch.nn.Sequential(torch.nn.Conv3d(in_planes, planes, kernel_size=ks, stride=stride,
#                                           padding=pad),
#                                 torch.nn.BatchNorm3d(planes),
#                                 torch.nn.GELU())
#         self.upsample_layer = torch.nn.Upsample(scale_factor=2, mode='nearest')
#     def forward(self, x):
#         # if self.add_coords:
#         #     x = add_coords(x)
#         out = self.c1(x)
#         if self.downsample:
#             out = torch.nn.functional.avg_pool3d(out, kernel_size=2, stride=2)
#         if self.upsample:
#             # out = torch.nn.Upsample(out, scale_factor=2, mode='nearest')
#             out = self.upsample_layer(out)
#         return(out)


#########################
#                       #
#    VISUALIZATIONS     #
#                       #
#########################

# def visualize_inference2D(gt, rec, template, filename = '/data/gt_overlay.png'):
#     '''
#     visualize overlay of ground truth neural substrate with infered reconstruction
#     '''
#     import numpy, matplotlib.pyplot as plt
#     plt.imshow(numpy.rot90(template[16,:,:],1), cmap = 'gray')
#     plt.imshow(numpy.where(gt>0,1,numpy.nan), cmap = 'hot')
#     plt.imshow(rec, cmap = 'plasma', alpha = 0.5)
#     plt.colorbar()
#     # ---- save figure ---- #
#     plt.savefig(filename)
#     plt.close()

def visualize_inference2D(gt, rec, template, filename = '/data/gt_overlay.png'):
    '''
    visualize overlay of ground truth neural substrate with infered reconstruction
    '''
    import numpy, matplotlib.pyplot as plt
    # plt.imshow(numpy.rot90(template[16,:,:],1), cmap = 'gray')
    plt.imshow(template, cmap = 'gray')
    plt.imshow(numpy.where(gt>0,1,numpy.nan), cmap = 'hot')
    plt.imshow(rec, cmap = 'plasma', alpha = 0.5)
    plt.colorbar()
    # ---- save figure ---- #
    plt.savefig(filename)
    plt.close()

# def visualize_inference(gt, rec, template, filename = '/data/gt_overlay.png'):
#     '''
#     visualize overlay of ground truth neural substrate with infered reconstruction
#     '''
#     import numpy, matplotlib.pyplot as plt
#     idx = numpy.mean(numpy.where(gt>0), axis =1).astype(int)
#     # ---- plot sagital view ---- #
#     plt.subplot(1,3,1)
#     plt.imshow(numpy.rot90(template[idx[0],:,:],1), cmap = 'gray', alpha = 0.5)
#     plt.imshow(numpy.where(numpy.rot90(gt[idx[0],:,:],1)>0,1,numpy.nan), cmap = 'pink', alpha = 0.5)
#     plt.imshow(numpy.rot90(rec[idx[0],:,:],1), cmap = 'plasma', alpha = 0.5)
#     # ---- plot coronal view ---- #
#     plt.subplot(1,3,2)
#     plt.imshow(numpy.rot90(template[:,idx[1],:],1), cmap = 'gray', alpha = 0.5)
#     plt.imshow(numpy.where(numpy.rot90(gt[:,idx[1],:],1)>0,1,numpy.nan), cmap = 'pink', alpha = 0.5)
#     plt.imshow(numpy.rot90(rec[:,idx[1],:],1), cmap = 'plasma', alpha = 0.5)
#     # ---- plot axial view ---- #
#     plt.subplot(1,3,3)
#     plt.imshow(numpy.rot90(template[:,:,idx[2]],1), cmap = 'gray', alpha = 0.5)
#     plt.imshow(numpy.where(numpy.rot90(gt[:,:,idx[2]],1)>0,1,numpy.nan), cmap = 'pink', alpha = 0.5)
#     plt.imshow(numpy.rot90(rec[:,:,idx[2]],1), cmap = 'plasma', alpha = 0.5)
#     plt.colorbar()
#     # ---- save figure ---- #
#     plt.savefig(filename)
#     plt.close()
