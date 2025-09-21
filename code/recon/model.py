#
# model.py
#
# this script contains class definitions for
# lesion mask reconstruction VAE
#
# these classes represent the first part of the pretraining
# of the joint-distribution VAE for lesion mask reconstruction
# and lesion deficit mapping.
#
#
#

import math
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from utils import *

# Define two globals
bce_fn = nn.BCELoss(reduction='none')
l2_fn = torch.nn.MSELoss(reduction = 'none')
Tensor = torch.cuda.FloatTensor


# ---- define VAE model ---- #

class VAERECON(nn.Module):
    def __init__(self, input_size, sd=16, z_dim=20, out_chans=1, in_chans=1):
        '''
        This is the VAE model that does the lesion mask reconstruction task.
        :param input_size:
        :param sd:
        :param z_dim:
        :param out_chans:
        :param in_chans:
        '''
        super(VAERECON, self).__init__()
        self.sd = sd
        self.z_dim = z_dim
        # self.half_z = z_dim // 2
        # Each layer reduces by a factor of 2, how many layers we need to get to latent space 2**3
        self.num_layers = int(math.log2(input_size)) - 1
        '''
        Encoder -- You'll probably need to tweak this to get the best results, GPU memory usage, etc.
        '''
        self.encoder_layers = nn.ModuleList()
        enc_sd = self.sd
        for l in range(self.num_layers):
            self.encoder_layers.append(SBlock(in_chans, enc_sd, downsample=True))
            in_chans = enc_sd
            if l < self.num_layers - 1:
                enc_sd *= 2
        # These are the dimensions of a fully connected latent at the end of the encoder
        # TODO: might not need to always be 2 cubed
        self.spatial_dims = input_size // (2 ** self.num_layers)
        self.dense_dims = self.spatial_dims ** 3 * (enc_sd)
        '''
        Parameters of the latent space
        '''
        self.mu = nn.Linear(self.dense_dims, z_dim)
        self.logvar = nn.Linear(self.dense_dims, z_dim)
        '''
        Decoders for the lesion reconstruction
        '''
        # self.decoder_inference = nn.ModuleList()
        self.decoder_reconstruction = nn.ModuleList()
        # self.decoder_inference.append(nn.Sequential(nn.Linear(self.half_z, self.dense_dims),
                                        #   nn.GELU()))
        self.decoder_reconstruction.append(nn.Sequential(nn.Linear(self.z_dim, self.dense_dims),
                                          nn.GELU()))
        dec_sd = enc_sd
        for l in range(self.num_layers):
            # self.decoder_inference.append(SBlock(dec_sd, dec_sd // 2, upsample=True))
            self.decoder_reconstruction.append(SBlock(dec_sd, dec_sd // 2, upsample=True))
            dec_sd = dec_sd // 2
        # Finish both decoders
        # self.decoder_inference.append(
        #     nn.Sequential(nn.Conv3d(dec_sd, int(dec_sd / 2), kernel_size=3, stride=1, padding=1),
        #                   nn.GELU(),
        #                   nn.Conv3d(int(dec_sd / 2), out_chans, kernel_size=1, stride=1, padding=0)
        #                 )
        # )
        self.decoder_reconstruction.append(
            nn.Sequential(nn.Conv3d(dec_sd, int(dec_sd / 2), kernel_size=3, stride=1, padding=1),
                          nn.GELU(),
                          nn.Conv3d(int(dec_sd / 2), out_chans, kernel_size=1, stride=1, padding=0)
                          )
        )
    def sampling(self, mu, log_var):
        '''
        Sample your latent from z ~ N(mean, scale)
        :param mu:
        :param log_var:
        :return:
        '''
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    def encoder(self, x):
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
        x = x.view(-1, self.dense_dims)
        return self.mu(x), self.logvar(x)
    # def decoder(self, x):
    #     x = self.decoder_inference[0](x)
    #     x = x.view(x.size(0), -1, self.spatial_dims, self.spatial_dims, self.spatial_dims)
    #     for dec_layer in self.decoder_inference[1:]:
    #         x = dec_layer(x)
    #     return x
    def rdecoder(self, x):
        x = self.decoder_reconstruction[0](x)
        x = x.view(x.size(0), -1, self.spatial_dims, self.spatial_dims, self.spatial_dims)
        for dec_layer in self.decoder_reconstruction[1:]:
            x = dec_layer(x)
        return x
    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        # mask_z = z[:, :self.half_z]
        # recon_z = z[:, self.half_z:]
        kl = torch.sum(0.5 * (-log_var + torch.exp(log_var) + mu ** 2 - 1), dim=1)
        return self.rdecoder(z), kl
        # return self.decoder(mask_z), self.rdecoder(recon_z), kl


class ModelWrapperRecon(nn.Module):
    def __init__(self, input_size, z_dim=50, start_dims=16, continuous=False, in_channels=1, out_channels=1, coords = False, lesion_threshold=None):
        '''
        A model wrapper around the VAE
        :param input_size:
        :param z_dim:
        :param start_dims:
        :param continuous:
        :param in_channels:
        :param out_channels:
        '''
        super().__init__()
        self.z_dim = z_dim
        self.start_dims = start_dims
        self.input_channels = in_channels
        self.output_channels = out_channels
        self.coordinate = coords
        self.lesion_threshold = lesion_threshold
        # validate if additional channels (coordinates, covariants, deficit_scores) improve lesion reconstruction
        # Nr of input channels - X, the coordinates, the covariants, and Y
        self.mask_model = VAERECON(input_size,
                              sd=start_dims,
                              z_dim=z_dim,
                              out_chans=self.output_channels,
                              in_chans=self.input_channels) #### ADJUSTED TO ACCOUNT FOR COVARIANTS
        self.continuous = continuous
    def forward(self, x,t=0.5, calibrate=False):
        '''
        If doing validation you will want to use the generated inference map to gauge the accuracy of the
        predictions
        :param x:
        :param val:
        :param provided_mask:
        :param provided_scale:
        :param t:
        :param calibrate:
        :return:
        '''
        b, c, h, w, d = x.shape
        # Add the label as a volume 
        # '''
        # add covariants (:= labels + covariants) as volume with N channels N:= number of covariants + 1 (deficit)
        # resulting in final input dimension of "x" as 1 (lesion mask channel) + 3 (coordinates) + N(labels) + N(covariants)
        # '''
        # my = y.view(-1, y.shape[2], 1, 1, 1).repeat(1, 1, h, w, d) #
        # my = y.view(-1, 1, 1, 1, 1).repeat(1, 1, h, w, d)
        # Add coordinates to the lesion if set to True
        if self.coordinate:
            x = add_coords(x)
            # x = torch.cat([coord_x], dim=1)
        # else:
            # x = torch.cat([x], dim = 1)
        recons, kl_m = self.mask_model(x)        
        # if val:
        #     # If doing validation use the masks calculated from the training data
        #     # Do a forward pass still so we can evaluate reconstruction quality and KL
        #     recons, kl_m = self.mask_model(x)
        #     # preds_mean = provided_mask
        #     # preds_scale = provided_scale
        # else:
        #     recons, kl_m = self.mask_model(x)
            # preds_mean = masks[:, 0].view(-1, 1, h, w, d)
            # preds_scale = masks[:, 1].view(-1, 1, h, w, d)
        # if calibrate:
        #     # If calibrating predictions, we want to find a thresholding quantile that achieves the best accuracy!
        #     flat_preds_a = recons.view(x.size(0), -1)
        #     qt = torch.quantile(flat_preds_a, t, dim=1).view(-1, 1, 1, 1, 1)
        #     recons = (recons > qt) * recons
        # The three outputs of our network -> Reconstructed lesion, Mean inference map and STD variance map
        recons = torch.sigmoid(recons)
        # If a lesion threshold is provided, binarise the reconstruction according to this and calculate the predictive loss with it
        # rather than with the original lesion
        if self.lesion_threshold:
            flat_preds_a = recons.view(x.size(0), -1)
            qt = torch.quantile(flat_preds_a, t, dim=1).view(-1, 1, 1, 1, 1)
            recons = (recons > qt).float()
        #     logits = torch.mean(bin_lesion * preds_mean, dim=(-4, -3, -2, -1)).view(-1, 1)
        #     # Standard deviation is currently between 0 and 1, but it can be larger or smaller
        #     scale = torch.mean(bin_lesion * preds_scale, dim=(-4, -3, -2, -1)).view(-1, 1).exp()
        # else:
        #     logits = torch.mean(x * preds_mean, dim=(-4, -3, -2, -1)).view(-1, 1)
        #     # Standard deviation is currently between 0 and 1, but it can be larger or smaller
        #     scale = torch.mean(x * preds_scale, dim=(-4, -3, -2, -1)).view(-1, 1).exp()
        # '''
        # Calculate log P(Y|X,M), i.e. the log-likelihood of our inference objective
        # '''
        # if self.continuous:
        #     # mask_ll = - D.Normal(logits, scale + 1e-5).log_prob(y).mean()
        #     mask_ll = torch.mean((logits - y) ** 2)
        # else:
        #     # Don't use STD on binary case because Bernoulli has no variance -> Beta distributions work well
        #     probabilities = torch.sigmoid(logits)
        #     mask_ll = bce_fn(probabilities, y).mean()
        '''
        Calculate log P(X|M), i.e. the log likelihood of our lesions 
        '''
        '''
        VALIDATE DIFFERENT LOSS FUNCTIONS, CHECK WITH GUILAUME
        '''
        recon_ll = torch.sum(bce_fn(recons, x), dim=(-3, -2, -1)).mean()
        # recon_ll = torch.sum(l2_fn(recons,x), dim=(-3, -2, -1)).mean()
        # preds = torch.mean(preds_mean, dim=0).view(1, 1, h, w, d)
        # mask_scale = torch.mean(preds_scale, dim=0).view(1, 1, h, w, d)
        # Calculate the accuracy of the predictions. If it is continuous, this is just MSE
        # if self.continuous:
        #     acc = mask_ll
        # else:
        #     quant_preds = (probabilities > 0.5).to(torch.float32)
        #     acc = torch.mean(torch.eq(quant_preds, y).float())
        '''
        The final loss is log P(Y| X, M) + log P(X|M) + D_KL[Q(M|X,Y) || P(M)]
        '''
        # loss = mask_ll + recon_ll + kl_m.mean()
        loss = recon_ll + kl_m.mean()
        ret_dict = dict(lesion_recon=recons,
                        kl=kl_m.mean(),
                        loss=loss,
                        recon_ll=recon_ll.mean()
                        )
        return ret_dict
    # def sample_masks(self, num_samples=400):
    #     '''
    #     Use this to sample the mean and STD masks from the latent space
    #     :param x:
    #     :param num_samples:
    #     :return:
    #     '''
    #     z = torch.randn(num_samples, self.z_dim).type(Tensor)
    #     preds = self.mask_model.decoder(z)
    #     mean_mask = torch.mean(preds[:, 0], dim=(0, 1))
    #     scale_mask = torch.mean(preds[:, 1], dim=(0, 1))
    #     return mean_mask, scale_mask
