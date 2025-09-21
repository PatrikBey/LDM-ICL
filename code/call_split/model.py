



import math

import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

# Define two globals
bce_fn = nn.BCELoss(reduction='none')
Tensor = torch.cuda.FloatTensor


def add_coords(x, just_coords=False):
    '''
    This just the Uber CoordConv method extended to 3D. Definitely use it on the input
    Using it on other layers of the model can be helpful, but it slows down training
    :param x:
    :param just_coords:
    :return:
    '''
    batch_size_shape, channel_in_shape, dim_y, dim_x = x.shape
    xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
    yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)
    xx_range = torch.arange(dim_y, dtype=torch.int32)
    yy_range = torch.arange(dim_x, dtype=torch.int32)
    xx_range = xx_range[None, None, :, None]
    yy_range = yy_range[None, None, :, None]
    xx_channel = torch.matmul(xx_range, xx_ones)
    yy_channel = torch.matmul(yy_range, yy_ones)
    # transpose y
    yy_channel = yy_channel.permute(0, 1, 3, 2)
    xx_channel = xx_channel.float() / (dim_y - 1)
    yy_channel = yy_channel.float() / (dim_x - 1)
    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1
    xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
    yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)
    xx_channel = xx_channel.cuda()
    yy_channel = yy_channel.cuda()    
    if just_coords:
        out = torch.cat([xx_channel, yy_channel], dim=1)
    else:
        out = torch.cat([x, xx_channel, yy_channel], dim=1)
    return out




class SBlock(nn.Module):
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
            in_planes += 3
        self.add_coords = add_coords
        self.c1 = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=ks, stride=stride,
                                          padding=pad),
                                nn.BatchNorm2d(planes),
                                nn.GELU())
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        if self.add_coords:
            x = add_coords(x)
        out = self.c1(x)
        if self.downsample:
            out = F.avg_pool2d(out, kernel_size=2, stride=2)
        if self.upsample:
            out = self.upsample_layer(out)
        return out


class VAE(nn.Module):
    def __init__(self, input_size, sd=16, z_dim=20, out_chans=1, in_chans=1, latent_split = False):
        '''
        This is the VAE model that does the lesion deficit mapping inference. It does two tasks with a single latent.
        First it produces the lesion-deficit map. Second it produces a reconstruction of the lesions.
        Both of these are necessary because we are modelling the joint distribution P(X,Y)
        There are many architectural improvements that will probably help get better accuracy, but this is a simple
        architecture that works even with little data. The more data you have, the more you might want to replace
        the Encoder and Decoder with something more complicated. Or even use a VDVAE
        Adding coordinates helps as well, but by default the models doesn't add them
        :param input_size:
        :param sd:
        :param z_dim:
        :param out_chans:
        :param in_chans:
        '''
        super(VAE, self).__init__()
        self.sd = sd
        self.z_dim = z_dim
        self.half_z = z_dim // 2
        # Each layer reduces by a factor of 2, how many layers we need to get to latent space 2**3
        self.num_layers = int(math.log2(input_size)) - 1
        self.latent_split = latent_split
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
        self.spatial_dims = input_size // (2 ** self.num_layers)
        self.dense_dims = self.spatial_dims ** 2 * (enc_sd)
        '''
        Parameters of the latent space
        '''
        self.mu = nn.Linear(self.dense_dims, z_dim)
        self.logvar = nn.Linear(self.dense_dims, z_dim)
        '''
        Decoders for the inference maps and lesion reconstructions
        '''
        self.decoder_inference = nn.ModuleList()
        self.decoder_reconstruction = nn.ModuleList()
        if self.latent_split:
            self.decoder_inference.append(nn.Sequential(nn.Linear(self.half_z, self.dense_dims),
                                            nn.GELU()))
            self.decoder_reconstruction.append(nn.Sequential(nn.Linear(self.half_z, self.dense_dims),
                                            nn.GELU()))
        else:
            self.decoder_inference.append(nn.Sequential(nn.Linear(self.z_dim, self.dense_dims),
                                            nn.GELU()))
            self.decoder_reconstruction.append(nn.Sequential(nn.Linear(self.z_dim, self.dense_dims),
                                            nn.GELU()))   
        dec_sd = enc_sd
        for l in range(self.num_layers):
            self.decoder_inference.append(SBlock(dec_sd, dec_sd // 2, upsample=True))
            self.decoder_reconstruction.append(SBlock(dec_sd, dec_sd // 2, upsample=True))
            dec_sd = dec_sd // 2
        # Finish both decoders
        self.decoder_inference.append(
            nn.Sequential(nn.Conv2d(dec_sd, int(dec_sd / 2), kernel_size=3, stride=1, padding=1),
                          nn.GELU(),
                          nn.Conv2d(int(dec_sd / 2), out_chans, kernel_size=1, stride=1, padding=0)
                        )
        )
        self.decoder_reconstruction.append(
            nn.Sequential(nn.Conv2d(dec_sd, int(dec_sd / 2), kernel_size=3, stride=1, padding=1),
                          nn.GELU(),
                          nn.Conv2d(int(dec_sd / 2), 1, kernel_size=1, stride=1, padding=0)
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
    def decoder(self, x):
        x = self.decoder_inference[0](x)
        x = x.view(x.size(0), -1, self.spatial_dims, self.spatial_dims)
        for dec_layer in self.decoder_inference[1:]:
            x = dec_layer(x)
        return x
    def rdecoder(self, x):
        x = self.decoder_reconstruction[0](x)
        x = x.view(x.size(0), -1, self.spatial_dims, self.spatial_dims)
        for dec_layer in self.decoder_reconstruction[1:]:
            x = dec_layer(x)
        return x
    def forward(self, x, y):
        mu, log_var = self.encoder(x)
        z = self.sampling(mu, log_var)
        if self.latent_split:
            mask_z = z[:, :self.half_z]
            recon_z = z[:, self.half_z:]
        else:
            mask_z = z
            recon_z = z
        kl = torch.sum(0.5 * (-log_var + torch.exp(log_var) + mu ** 2 - 1), dim=1)
        return self.decoder(mask_z), self.rdecoder(recon_z), kl


class ModelWrapper(nn.Module):
    def __init__(self, input_size, z_dim=128, start_dims=16, continuous=False, aci = True, latent_split = True, template = None):
        '''
        A model wrapper around the VAE
        :param input_size:
        :param z_dim:
        :param start_dims:
        :param continuous:
        :param aci:
        '''
        super().__init__()
        self.z_dim = z_dim
        self.start_dims = start_dims
        self.latent_split = latent_split
        # 4 input channels - X, the coordinates, and Y
        # 2 output channels - The mean and the variance of the inference maps
        self.mask_model = VAE(input_size,
                              sd=start_dims,
                              z_dim=z_dim,
                              out_chans=2,
                              in_chans=4,
                              latent_split=self.latent_split)
        self.continuous = continuous
        self.aci = aci
        if self.aci:
            self.template=template
    def forward(self, x, y, val=False, provided_mask=None,
                provided_scale=None, t=0.5, calibrate=False):
        '''
        If doing validation you will want to use the generated inference map to gauge the accuracy of the
        predictions
        :param x:
        :param y:
        :param val:
        :param provided_mask:
        :param provided_scale:
        :param t:
        :param calibrate:
        :return:
        '''
        import numpy as np
        b, c, h, w = x.shape
        # Add coordinates to the lesion
        coord_x = add_coords(x)
        # Add the label as a volume
        my = y.view(-1, 1, 1, 1).repeat(1, 1, h, w)
        # ---- anatomically constrained inference ---- #
        if self.aci:
            self.aci_mask = np.repeat(self.template[None,:],b,axis = 0)
            self.aci_mask = np.expand_dims(self.aci_mask,axis = 1)
            self.aci_mask = torch.tensor(self.aci_mask, device = x.device)
            my = my * self.aci_mask
        coord_x = torch.cat([coord_x, my], dim=1)
        if val:
            # If doing validation use the masks calculated from the training data
            # Do a forward pass still so we can evaluate reconstruction quality and KL
            masks, recons, kl_m = self.mask_model(coord_x, y)
            preds_mean = provided_mask
            preds_scale = provided_scale
        else:
            masks, recons, kl_m = self.mask_model(coord_x, y)
            preds_mean = masks[:, 0].view(-1, 1, h, w)
            preds_scale = masks[:, 1].view(-1, 1, h, w)
        if calibrate:
            # If calibrating predictions, we want to find a thresholding quantile that achieves the best accuracy!
            flat_preds_a = preds_mean.view(x.size(0), -1)
            qt = torch.quantile(flat_preds_a, t, dim=1).view(-1, 1, 1, 1)
            preds_mean = (preds_mean > qt) * preds_mean
        # The three outputs of our network -> Reconstructed lesion, Mean inference map and STD variance map
        recons = torch.sigmoid(recons)
        logits = torch.mean(x * preds_mean, dim=(-3, -2, -1)).view(-1, 1)
        # Standard deviation is currently between 0 and 1, but it can be larger or smaller
        scale = torch.mean(x * preds_scale, dim=(-3, -2, -1)).view(-1, 1).exp()
        '''
        Calculate log P(Y|X,M), i.e. the log-likelihood of our inference objective
        '''
        if self.continuous:
            # mask_ll = - D.Normal(logits, scale + 1e-5).log_prob(y).mean()
            mask_ll = torch.mean((logits - y) ** 2)
        else:
            # Don't use STD on binary case because Bernoulli has no variance -> Beta distributions work well
            probabilities = torch.sigmoid(logits)
            mask_ll = bce_fn(probabilities, y).mean()
        '''
        Calculate log P(X|M), i.e. the log likelihood of our lesions 
        '''
        recon_ll = torch.sum(bce_fn(recons, x), dim=(-2, -1)).mean()
        preds = torch.mean(preds_mean, dim=0).view(1, 1, h, w)
        mask_scale = torch.mean(preds_scale, dim=0).view(1, 1, h, w)
        # Calculate the accuracy of the predictions. If it is continuous, this is just MSE
        if self.continuous:
            acc = mask_ll
        else:
            quant_preds = (probabilities > 0.5).to(torch.float32)
            acc = torch.mean(torch.eq(quant_preds, y).float())
        '''
        The final loss is log P(Y| X, M) + log P(X|M) + D_KL[Q(M|X,Y) || P(M)]
        '''
        loss = mask_ll + recon_ll + kl_m.mean()
        # weightsum = mask_ll + recon_ll + kl_m.mean()
        # weights = [mask_ll / weightsum, recon_ll / weightsum, kl_m.mean() / weightsum]  
        # mask_ll = 1/mask_ll*(1-weights[0])
        # recon_ll = 1/recon_ll*(1-weights[1])
        # kl = 1/kl_m.mean()*(1-weights[2])
        # loss = mask_ll + recon_ll + kl
        # loss = mask_ll*(1-weights[0])*0.5 + recon_ll*(1-weights[1])*2 + kl_m.mean()*(1-weights[2])
        ret_dict = dict(mean_mask=preds,
                        mask_scale=mask_scale,
                        mask_ll=mask_ll.mean(),
                        # kl=kl,
                        kl=kl_m.mean(),
                        loss=loss, acc=acc,
                        recon_ll=recon_ll.mean(),
                        lesion_recon = recons
                        )
        return ret_dict
    def sample_masks(self, num_samples=400):
        '''
        Use this to sample the mean and STD masks from the latent space
        :param x:
        :param num_samples:
        :return:
        '''
        z = torch.randn(num_samples, self.z_dim).type(Tensor)
        preds = self.mask_model.decoder(z)
        mean_mask = torch.mean(preds[:, 0], dim=(0, 1))
        scale_mask = torch.mean(preds[:, 1], dim=(0, 1))
        return mean_mask, scale_mask

