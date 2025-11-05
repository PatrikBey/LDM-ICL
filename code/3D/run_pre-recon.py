#
#
#
# This script runs the lesion reconstruction
# pretraining for the 3D LDM-ICL model
#


import scipy.io, os, json
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from mpl_toolkits.axes_grid1 import ImageGrid
# from sklearn.model_selection import train_test_split
import sklearn

from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim

import shutil
import torch as tc

os.chdir('3D')
from model import *
from utils import log_msg, get_variable, get_device, LesionDataset, visualize_inference3D, count_parameters

#########################################
#                                       #
#              PARSE INPUT              #
#                                       #
#########################################

log_msg('START | Running Deep Variational Lesion Reconstruction')

# TEMPLATEDIR=os.getenv("TEMPLATEDIR")

Path='/data'
# Path='/mnt/h/DLDM/3D'

# ---- set output directory ---- #
if os.path.isdir('/data'):
    out_dir = Path+'/pre-recon'
    os.makedirs(out_dir, exist_ok=True)
else:
    out_dir = Path


# ---- load lesion masks ---- #
lesions= np.load(os.path.join(Path,'lesions','Synthetic-masks-recon-10K.npy')).astype(np.int32)
# lesions= np.load(os.path.join(Path,'lesions','Ischaemic_lesions_3D.npy')).astype(np.int32)

# --- ensure lesion masks are binary --- #
if len(np.unique(lesions)) > 2:
    lesions = np.where(lesions>0,1,0)


# ---- create aggreate lesion mask ---- #
aggregate = np.sum(lesions, axis=0)

# nibabel.save(nibabel.Nifti1Image(aggregate.astype(np.float32), np.eye(4)), out_dir+'/lesions_aggregate.nii.gz')

mni_brain = nibabel.load(os.path.join(Path,'templates','MNI152_64_brain.nii.gz')).get_fdata()

visualize_inference3D(aggregate, aggregate, mni_brain, out_dir + '/lesions_aggregate.png')




##################################
#                                #
#       TRAIN / TEST SPLIT       #
#                                #
##################################

# ---- expand lesion array dimensionality ---- #
if lesions.ndim == 4:
    lesions = np.expand_dims(lesions, axis=1)


# ---- single 10% train / test split ---- #
train_data, val_data = sklearn.model_selection.train_test_split(lesions, test_size=0.1)

# # ---- split test into validation / calibration 50% ---- #
# val_data, cal_data = sklearn.model_selection.train_test_split(vc_data, test_size=0.5)
# val_data = vc_data




##################################
#                                #
# PREPARE DATA LOADER / BATCHES  #
#                                #
##################################


batch_size = 256


# CREATE DATA LOADERS
# TRAINING
dataset = LesionDataset(data=train_data)

train_loader = DataLoader(dataset, 
                          batch_size=batch_size, 
                          drop_last=False,
                          shuffle=True, 
                          num_workers=0, 
                          pin_memory=True)

# VALIDATION
val_dataset = LesionDataset(data=val_data)

val_loader = DataLoader(val_dataset, 
                        batch_size=batch_size, 
                        drop_last=False,
                        shuffle=True,
                        num_workers=0, 
                        pin_memory=True)

# # CALIBRATION
# cal_dataset = LesionDataset(data=cal_data)
# cal_loader = DataLoader(cal_dataset, 
#                         batch_size=batch_size, 
#                         drop_last=False,
#                         shuffle=True,
#                         num_workers=0, 
#                         pin_memory=True)


device = get_device()





##################################
#                                #
#           BUILD MODEL          #
#                                #
##################################


# FIRST ELEMENT IS THE SIZE OF THE VOLUMES IN VOXELS
# CURRENTLY CUBE IMAGES ARE REQUIRED - PAD WITH 0 IF YOUR IMAGE IS NOT CUBED
INPUT_SIZE = dataset[0].shape[-1]

CONTINUOUS = False
Z_DIM = 40 # use 40 for no-latent split LDM, 20 for latent split LDM
EPOCHS = 500
INITIAL_CONV_KERNELS = 16
L2_REG = 1e-4
LR = 5e-3

Tensor = torch.cuda.FloatTensor

model = ModelWrapperRecon(INPUT_SIZE,
                     z_dim=Z_DIM,
                     start_dims=INITIAL_CONV_KERNELS,
                     continuous=CONTINUOUS,
                     in_channels=1, # only lesion mask input channel
                     lesion_threshold=False).to(device)


if model.continuous:
    log_msg('UPDATE | using continuous model')


# Other optimisers work as well, Adamax is quite stable though
optimizer = optim.Adamax(model.parameters(),
                         weight_decay=L2_REG,
                         lr=LR)

log_msg('UPDATE | model parameter count: {}'.format(count_parameters(model)))
log_msg(f'UPDATE | epochs : {EPOCHS}')


##################################
#                                #
#           TRAIN MODEL          #
#                                #
##################################




best_loss = 1e30
best_acc = 0
# best_lk = 1e30
global_step = 0

training_losses = []
validation_losses = []

dims = dataset[0].shape[1:]

# ---- initial validation loss ---- #
val_acc = 0
accuracy_acc = 0
loss_acc = 0
likelihood_acc = 0
kld_acc = 0
recon_acc = 0

with torch.no_grad():
    for x in val_loader:
        x = x.type(Tensor).to(device)
        ret_dict = model(x)
        loss_acc += ret_dict['loss'].mean().item()
        val_acc += 1
        kld_acc += ret_dict['kl'].item()
        recon_acc += ret_dict['recon_ll'].item()

validation_losses.append(ret_dict['loss'].mean().item())

with torch.no_grad():
    for x in train_loader:
        x = x.type(Tensor).to(device)
        ret_dict = model(x)
        loss_acc += ret_dict['loss'].mean().item()
        val_acc += 1
        kld_acc += ret_dict['kl'].item()
        recon_acc += ret_dict['recon_ll'].item()

training_losses.append(ret_dict['loss'].mean().item())


for epoch in range(EPOCHS):
    model.zero_grad()
    train_acc = 0
    t_epoch_loss = 0
    for x in train_loader:
        optimizer.zero_grad()
        x = x.type(Tensor).to(device)
        ret_dict = model(x)
        loss = ret_dict['loss'].mean()
        t_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += 1
        global_step += 1
    if epoch % 10 == 0:
        recon = ret_dict['lesion_recon'].cpu().data.numpy()[10,:,:,:].reshape(dims)
        mask = x.cpu().data.numpy()[10,:,:,:].reshape(dims)
        visualize_inference3D(mask, recon, mni_brain, os.path.join(out_dir,f'reconstruction-train-epoch_{epoch}.png') )
    training_losses.append(t_epoch_loss / train_acc)
    val_acc = 0
    accuracy_acc = 0
    loss_acc = 0
    likelihood_acc = 0
    kld_acc = 0
    recon_acc = 0
    with torch.no_grad():
        for x in val_loader:
            x = x.type(Tensor).to(device)
            ret_dict = model(x)
            loss_acc += ret_dict['loss'].mean().item()
            val_acc += 1
            kld_acc += ret_dict['kl'].item()
            recon_acc += ret_dict['recon_ll'].item()
    loss = loss_acc / val_acc
    validation_losses.append(loss)
    acc = round(accuracy_acc / val_acc, 4)
    kl = round(kld_acc / val_acc, 3)
    rec = recon_acc / val_acc
    if loss < best_loss:
        best_loss = loss
        best_acc = acc
        best_recon = recon_acc
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(out_dir,'pre-recon_vae.pth'))
        # log_msg(f'UPDATE | Saving current model')
    if epoch % 10 == 0:
        log_msg(f'UPDATE | Best: {best_loss}, epoch: {best_epoch}')
        recon = ret_dict['lesion_recon'].cpu().data.numpy()[10,:,:,:].reshape(dims)
        mask = x.cpu().data.numpy()[10,:,:,:].reshape(dims)
        visualize_inference3D(mask, recon, mni_brain, os.path.join(out_dir,f'reconstruction-val-epoch_{epoch}.png') )


plt.plot(np.log(training_losses), label='training loss')
plt.plot(np.log(validation_losses), label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Log-loss')
plt.legend()
plt.savefig(os.path.join(out_dir,'loss_curve.png'))
plt.close()


for th in [0.25,0.5,0.75,0.9,0.95,0.975,0.99]:
    tmp = ret_dict['lesion_recon'].cpu().data.numpy()[10,:,:,:].reshape(dims)
    testing = np.where(tmp>np.quantile(tmp,th),1,0)
    mask = x.cpu().data.numpy()[10,:,:,:].reshape(dims)
    visualize_inference3D(mask, testing, mni_brain, os.path.join(out_dir, f'Reconstruction_threshold_{th}.png'))



# visualize_inference3D(mask, recon, mni_brain, os.path.join(out_dir,f'reconstruction-val-final.png') )

# log_msg('FINISHED | Running Deep Variational Lesion Reconstruction')

# for th in [0.25,0.5,0.75,0.9,0.95,0.975,0.99]:
#     tmp = ret_dict['lesion_recon'].cpu().data.numpy()[10,:,:,:].reshape(dims)
#     testing = np.where(tmp>np.quantile(tmp,th),1,0)
#     dice_3D(mask,testing)


