#
# model.py (2D version)
#
# This script contains class definitions for
# lesion mask reconstruction VAE (2D version)
#

import math
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from utils import *

# Define two globals
bce_fn = nn.BCELoss(reduction='none')
l2_fn = torch.nn.MSELoss(reduction='none')
Tensor = torch.cuda.FloatTensor

# ---- define VAE model ---- #

class VAERECON(nn.Module):
    def __init__(self, input_size, sd=16, z_dim=20, out_chans=1, in_chans=1):
        '''
        This is the VAE model that does the lesion mask reconstruction task (2D version).
        '''
        super(VAERECON, self).__init__()
        self.sd = sd
        self.z_dim = z_dim
        self.num_layers = int(math.log2(input_size)) - 1
        self.encoder_layers = nn.ModuleList()
        enc_sd = self.sd
        for l in range(self.num_layers):
            self.encoder_layers.append(SBlock(in_chans, enc_sd, downsample=True))
            in_chans = enc_sd
            if l < self.num_layers - 1:
                enc_sd *= 2
        self.spatial_dims = input_size // (2 ** self.num_layers)
        self.dense_dims = self.spatial_dims ** 2 * (enc_sd)
        self.mu = nn.Linear(self.dense_dims, z_dim)
        self.logvar = nn.Linear(self.dense_dims, z_dim)
        self.decoder_reconstruction = nn.ModuleList()
        self.decoder_reconstruction.append(nn.Sequential(nn.Linear(self.z_dim, self.dense_dims),
                                                        nn.GELU()))
        dec_sd = enc_sd
        for l in range(self.num_layers):
            self.decoder_reconstruction.append(SBlock(dec_sd, dec_sd // 2, upsample=True))
            dec_sd = dec_sd // 2
        self.decoder_reconstruction.append(
            nn.Sequential(
                nn.Conv2d(dec_sd, int(dec_sd / 2), kernel_size=3, stride=1, padding=1),
                nn.GELU(),
                nn.Conv2d(int(dec_sd / 2), out_chans, kernel_size=1, stride=1, padding=0)
            )
        )
    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def encoder(self, x):
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
        x = x.view(-1, self.dense_dims)
        return self.mu(x), self.logvar(x)
    def rdecoder(self, x):
        x = self.decoder_reconstruction[0](x)
        x = x.view(x.size(0), -1, self.spatial_dims, self.spatial_dims)
        for dec_layer in self.decoder_reconstruction[1:]:
            x = dec_layer(x)
        return x
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        kl = torch.sum(0.5 * (-log_var + torch.exp(log_var) + mu ** 2 - 1), dim=1)
        return self.rdecoder(z), kl


class ModelWrapperRecon(nn.Module):
    def __init__(self, input_size, z_dim=50, start_dims=16, continuous=False, in_channels=1, out_channels=1, coords=False, lesion_threshold=None):
        '''
        A model wrapper around the VAE (2D version)
        '''
        super().__init__()
        self.z_dim = z_dim
        self.start_dims = start_dims
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.coordinate = coords
        self.lesion_threshold = lesion_threshold
        self.mask_model = VAERECON(input_size,
                              sd=start_dims,
                              z_dim=z_dim,
                              out_chans=self.output_channels,
                              in_chans=self.input_channels)
        self.continuous = continuous
    def forward(self, x, t=0.5, calibrate=False):
        b, c, h, w = x.shape
        if self.coordinate:
            x = add_coords(x)
        recons, kl_m = self.mask_model(x)
        recons = torch.sigmoid(recons)
        if self.lesion_threshold:
            flat_preds_a = recons.view(x.size(0), -1)
            qt = torch.quantile(flat_preds_a, t, dim=1).view(-1, 1, 1, 1)
            recons = (recons > qt).float()
        recon_ll = torch.sum(bce_fn(recons, x), dim=(-2, -1)).mean()
        loss = recon_ll + kl_m.mean()
        ret_dict = dict(lesion_recon=recons,
                        kl=kl_m.mean(),
                        loss=loss,
                        recon_ll=recon_ll.mean()
                        )
        return ret_dict