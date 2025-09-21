#
#
# base_model_run_cov.py
#
# authors :
# 1. Pombo, Guilaume
# 2. Bey, Patrik (containerization)
#
#
#
#
# This script runs the base model as given by the Jupyter notebook
# from the original preprint of Pombo et al. (2023) performing
# deep variational lesion deficit mapping for example toy data.
#
# This script contains the updates to handle multidimensional
# representations of deficit scores to incorporate covariants.
#
#
#
######



######
# PYTHON SETUP
######

import sklearn.model_selection
import numpy, os, argparse, torch, datetime, torch.optim as optim, numpy, matplotlib.pyplot as plt, scipy.stats, math, nibabel, random, sklearn #, cv2

os.chdir('2D/recon')


from model2D import *

from utils import *


from torch.utils.data import Dataset, DataLoader


#########################################
#                                       #
#              PARSE INPUT              #
#                                       #
#########################################

log_msg('START | Running Deep Variational Lesion Deficit Mapping.')

Path=os.getenv("TEMPLATEDIR")

if os.path.isdir('/data'):
    out_dir = '/data/pretrain_20K'
    os.makedirs(out_dir, exist_ok=True)
else:
    out_dir = Path


# # ---- load lesion masks ---- #
lesions= numpy.load(os.path.join(Path,'validation','20000_lesions_2D.npy'))
# lesions= numpy.load(os.path.join('/data','10K_lesions.npy'))

# lesions1 = numpy.expand_dims(lesions1, axis=1)

# lesions2 = numpy.load(os.path.join(Path,'validation','labelsnew.npy'))

# lesions = numpy.concatenate((lesions1, lesions2), axis=0)

# ---- load template brain ---- #
# template_brain = numpy.load(os.path.join(Path,'validation','mni_brain_32.npy'))
template_brain = np.rot90(np.sum(np.load(os.path.join(Path,'validation','mni_brain_32.npy')), axis = 0),1)

aggregate = np.sum(lesions, axis=0)
visualize_inference2D(aggregate, aggregate, template_brain, out_dir + '/lesions_aggregate.png')

##################################
#                                #
#       TRAIN / TEST SPLIT       #
#                                #
##################################

# ---- expand lesion array dimensionality ---- #
if lesions.ndim == 3:
    lesions = numpy.expand_dims(lesions, axis=1)


'''
TESTING DEFICIT SCORE / COVARIANT IMPACT ON LESION RECONSTRUCTION

1. DONT INCLUDE EITHER
>> No labels beside lesion masks
'''

# ---- single 10% train / test split ---- #
train_data, vc_data = sklearn.model_selection.train_test_split(lesions, test_size=0.1)

# ---- split test into validation / calibration 50% ---- #
val_data, cal_data = sklearn.model_selection.train_test_split(vc_data, test_size=0.5)



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

# CALIBRATION
cal_dataset = LesionDataset(data=cal_data)
cal_loader = DataLoader(cal_dataset, 
                        batch_size=batch_size, 
                        drop_last=False,
                        shuffle=True,
                        num_workers=0, 
                        pin_memory=True)


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
Z_DIM = 20
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
        visualize_inference2D(mask, recon, template_brain, os.path.join(out_dir,f'reconstruction-train-epoch_{epoch}.png') )
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
    # log_msg(f'UPDATE | Epoch: {epoch}, KL: {kl}, recon likelihood: {rec}')
    if loss < best_loss:
        best_loss = loss
        best_acc = acc
        best_recon = recon_acc
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(out_dir,'recon_vae.pth'))
        # log_msg(f'UPDATE | Saving current model')
    if epoch % 10 == 0:
        log_msg(f'UPDATE | Best: {best_loss}, epoch: {best_epoch}')
        recon = ret_dict['lesion_recon'].cpu().data.numpy()[10,:,:,:].reshape(dims)
        mask = x.cpu().data.numpy()[10,:,:,:].reshape(dims)
        visualize_inference2D(mask, recon, template_brain, os.path.join(out_dir,f'reconstruction-val-epoch_{epoch}.png') )
        # subtitle_fontsize = 30

log_msg('FINISHED | Running Deep Variational Lesion Deficit Mapping.')

