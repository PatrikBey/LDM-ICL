#########################################################################
#                                      ###     ###    #######   ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                      ###     ###   ###        ###     #
#                                       #########     #######   #########
#                                                                       #
# DEEP LESION DEFICIT MAPPING | FINE-TUNING                             #
#                                                                       #
# This script performs the second stage of deep lesion deficit mapping  #
# by fine-tuning the DLDM model after reconstruction pretraining.       #
#                                                                       #
# Author: Patrik Bey, patrik.bey@ucl.ac.uk                              #
#                                                                       #
# last update: 2025/11/03                                               #
#                                                                       #
# INPUT:                                                                #
#  - lesion masks | pre-tune-5K.npy                                     #
#  - substrate   | AAL3_roi_1.npy                                       #
#  - deficit scores | deficit_scores.npy                                #
#                                                                       #
# OUTPUT:                                                               #
#  - fine-tuned model weights | pre-tune_vae.pth                        #
#  - deficit predictions | substrate_prediction.npy                     #
#  - training / validation performance arrays & plots                   #
#                                                                       #
#########################################################################

import sys,scipy.io, os, json, numpy as np, matplotlib.pyplot as plt, shutil, torch, torch.optim as optim, nibabel

# from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# ---- import model classes ---- #
os.chdir('3D')
from model import *
from utils import log_msg, get_variable, get_device, DeficitDataset, visualize_inference3D, count_parameters, vec_dice, dice_3D, get_deficit

log_msg("START | running deep lesion deficit mapping")

#########################################
#                                       #
#              PARSE INPUT              #
#                                       #
#########################################

log_msg("UPDATE | parsing input variables")

# ---- set global variables ---- #
Path = '/data'
# Path = '/mnt/h/DLDM/3D'
device = get_device()
Tensor = torch.cuda.FloatTensor

# ---- lesion subset count ---- #
n_lesions = get_variable('N_LESIONS')
# set N for initial testing purposes
if n_lesions:
    n_lesions = int(n_lesions)

# ---- output directory ---- #

out_dir = get_variable('OUTDIR')

if not out_dir:
    out_dir = os.path.join(Path, 'pre-tune')
else:
    out_dir = os.path.join(Path, out_dir)

if not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)

log_msg(f'UPDATE | output directory: {out_dir}')

# ---- pretraining  ---- #
pretraining = get_variable('PRETRAINING')

if pretraining:
    pretraining = eval(pretraining)
    if pretraining:
        log_msg(f'UPDATE | utilizing pretrained model')
else:
    pretraining=True
    log_msg(f'UPDATE | utilizing pretrained model')

# ---- pretraining model weights ---- #

model_path = get_variable('MODEL_PATH')
if not model_path: 
    model_path = os.path.join(Path,'pre-recon','pre-recon_vae.pth')

if pretraining:
    log_msg(f'UPDATE | using pretrained model weights: {model_path}')

# ---- anatomically constrained inference  ---- #
aci = get_variable('ACI')
if aci:
    aci = eval(aci)
    if aci:
        log_msg(f'UPDATE | utilizing anatomically constrained model')

# ---- lesion sets ---- #

train_lesion_type = get_variable('TRAIN_LESION_TYPE')

if not train_lesion_type:
    train_lesion_type = 'Synthetic-masks-tune-5K.npy'

log_msg(f'UPDATE | training lesion type: {train_lesion_type}')


# ---- substrate ---- #

substrate_type = get_variable('SUBSTRATE_TYPE')

if not substrate_type:
    substrate_type = 'AAL3_roi_1.npy'

log_msg(f'UPDATE | using substrate: {substrate_type}')


# ---- deficit scores ---- #

deficits_train = get_variable('DEFICITS_TRAIN')

if not deficits_train:
    deficits_train = 'overlap-ratio-noisy'

log_msg(f'UPDATE | synthetic functional deficit: {deficits_train}')


#########################################
#                                       #
#               DATA I/O                #
#                                       #
#########################################

# ---- lesions ---- #

try:
    lesions = np.load(os.path.join(Path, 'lesions', train_lesion_type))    
except:
    log_msg(f'ERROR | lesion file not found: {os.path.join(Path, "lesions", train_lesion_type)}')
    sys.exit(1)


# ---- deficit scores ---- #

try:
    deficits = np.load(os.path.join(Path, 'deficits', f'{deficits_train}_{train_lesion_type}'))
except:
    log_msg(f'ERROR | deficit scores file not found: {os.path.join(Path, "deficits", f"{deficits_train}_scores.npy")}')
    sys.exit(1)


# ---- limit number of lesions for testing ---- #
if n_lesions:
    lesions = lesions[:n_lesions,:]
    deficits = deficits[:n_lesions]

# ---- neural substrate ---- #

try:
    substrate = np.load(os.path.join(Path,'substrates',substrate_type))
except:
    log_msg(f'ERROR | substrate file not found: {substrate_type}')
    sys.exit(1)

# ---- ensure binary masks ---- #
lesions = np.where(lesions>0,1,0)
substrate = np.where(substrate>0,1,0)



# ---- load pretrained weights ---- #
if pretraining:
    pretrain_dict = torch.load(model_path, map_location=device)
    pretrain_keys = list(pretrain_dict.keys())
    # ---- get model parameters from pretrained model ---- #
    PRE_Z_DIM = pretrain_dict['mask_model.mu.weight'].shape[0]
    PRE_INITIAL_CONV_KERNELS = pretrain_dict[pretrain_keys[0]].shape[0]
    log_msg(f'UPDATE | loaded pretrained model weights from: {model_path}')



#########################################
#                                       #
#              PREPROCESSING            #
#                                       #
#########################################

# ---- split training / testing sets ---- #
train_lesions, test_lesions, train_labels, test_labels = train_test_split(lesions, deficits,test_size=0.1, random_state=42)

sum_check = np.sum(train_lesions, axis=tuple(np.arange(1, train_lesions.ndim)))
empty_lesion = np.where(sum_check==0)
train_lesions = np.delete(train_lesions, empty_lesion, axis=0)
log_msg(f'UPDATE | number of empty training lesions removed: {len(empty_lesion[0])}')

sum_check = np.sum(test_lesions, axis=tuple(np.arange(1, test_lesions.ndim)))
empty_lesion = np.where(sum_check==0)
test_lesions = np.delete(test_lesions, empty_lesion, axis=0)
log_msg(f'UPDATE | number of empty validation lesions removed: {len(empty_lesion[0])}')

log_msg(f'UPDATE | number of training lesions: {train_lesions.shape[0]}')
log_msg(f'UPDATE | number of validation lesions: {test_lesions.shape[0]}')

# ---- load template brain ---- #
# template_brain = np.load(os.path.join(Path,'validation','MNI152_T1_32.npy'))
template_brain = nibabel.load(os.path.join(Path,'templates','MNI152_64_brain.nii.gz')).get_fdata()
log_msg(f'UPDATE | loaded template brain from: {os.path.join(Path,"templates","MNI152_64_brain.nii.gz")}')


# ---- visualize lesion aggregates ---- #
train_aggregate = np.sum(train_lesions, axis=0)
test_aggregate =  np.sum(test_lesions, axis=0)

empty = np.ones(train_aggregate.shape)
visualize_inference3D(empty, train_aggregate, template_brain, os.path.join(out_dir, f'lesion_aggregate_train.png'))
visualize_inference3D(empty, test_aggregate, template_brain, os.path.join(out_dir, f'lesion_aggregate_test.png'))

#########################################
#                                       #
#           MODEL PARAMETERS            #
#                                       #
#########################################

params = ['CONTINUOUS', 'Z_DIM','EPOCHS','INITIAL_CONV_KERNELS','L2_REG','LR', 'LATENT_SPLIT']


# ---- load default parameters ---- #
with open(os.path.join(Path,'templates','model-params-defaults.json')) as f:
    model_params = json.load(f)
    print(model_params)

# ---- update with user parameters ---- #
for p in params:
    locals()[p] = get_variable(p)
    if locals()[p]:
        model_params[p] = eval(locals()[p])

# ---- update with pretraining parameters ---- #
if pretraining:
    model_params['Z_DIM'] = PRE_Z_DIM
    model_params['INITIAL_CONV_KERNELS'] = PRE_INITIAL_CONV_KERNELS


model_params['INPUT_SIZE'] = train_lesions.shape[-1]
# model_params['CONTINUOUS'] = True

log_msg('UPDATE | using model parameters:')
for p in model_params.keys():
    log_msg(f'UPDATE | {p}: {model_params[p]}')

# ---- save final model parameters ---- #
with open(os.path.join(out_dir,'model_parameters.json'), "w") as f:
        json.dump(model_params, f, indent=4)

#########################################
#                                       #
#               DATASETS                #
#                                       #
#########################################

# ---- add color channel if missing---- #
if train_lesions.ndim < 5:
    train_lesions = np.expand_dims(train_lesions, axis=1)

if test_lesions.ndim < 5:
    test_lesions = np.expand_dims(test_lesions, axis=1)

# ---- determine batch size ---- #
n_samples = train_lesions.shape[0]

if n_samples > 511:
    batch_size = 128
else:
    batch_size = int(np.round(n_samples // 10,0))

log_msg(f'UPDATE | batch size: {batch_size}')

# ---- prepare training dataset u---- #

train_dataset = DeficitDataset(data=train_lesions, labels=train_labels)
train_loaders = DataLoader(train_dataset, 
                           batch_size=batch_size, 
                           drop_last=False,
                           shuffle=True, 
                           num_workers=0, 
                           pin_memory=True)

# ---- prepare testing dataset---- #

test_dataset = DeficitDataset(data=test_lesions, labels=test_labels)
test_loader = DataLoader(test_dataset, 
                        batch_size=batch_size, 
                        drop_last=False,
                        shuffle=True, 
                        num_workers=0, 
                        pin_memory=True)


#########################################
#                                       #
#              VAE MODEL                #
#                                       #
#########################################


torch.manual_seed(42)

# ---- define model ---- #
model = ModelWrapper(model_params['INPUT_SIZE'],
                     z_dim=model_params['Z_DIM'],
                     start_dims=model_params['INITIAL_CONV_KERNELS'],
                     continuous=model_params['CONTINUOUS'],
                     aci=aci,
                     template = np.where(template_brain>0,1,0),
                     latent_split=eval(model_params['LATENT_SPLIT'])).to(device)


# ---- define Adamax optimizer ---- #
optimizer = optim.Adamax(model.parameters(),
                         weight_decay=model_params['L2_REG'],
                         lr=model_params['LR'])

log_msg('UPDATE | model paramter count: {}'.format(count_parameters(model)))




#########################################
#                                       #
#             PRETRAINING               #
#                                       #
#########################################

if pretraining:
    model_keys = list(model.state_dict().keys())
    # ---- MAP ENCODER WEIGHTS ---- #
    dims = model.state_dict()[model_keys[0]].shape
    for i in range(dims[1]):
        model.state_dict()[model_keys[0]][:,i,:,:] = model.state_dict()[model_keys[0]][:,i,:,:].copy_(pretrain_dict[pretrain_keys[0]][:,0,:,:])
    log_msg(f'UPDATE | mapped: {model_keys[0]}')
    for k in pretrain_keys[1:]:
        if 'encoder' in k:
            model.state_dict()[k] = model.state_dict()[k].copy_(pretrain_dict[k])
            log_msg(f'UPDATE | mapped: {k}')
    # ---- MAP RECONSTRUCTION WEIGHTS ---- #
    for k in pretrain_keys[1:]:
        if 'decoder_reconstruction' in k:
            model.state_dict()[k] = model.state_dict()[k].copy_(pretrain_dict[k])
            log_msg(f'UPDATE | mapped: {k}')
    # ---- FREEZE RECONSTRUCTION WEIGHTS ---- #
    # for layer in model.mask_model.decoder_reconstruction.parameters():
    #     layer.requires_grad = False
    # log_msg(f'UPDATE | frozen reconstruction decoder weights')
    # ---- FREEZE ENCODER WEIGHTS ---- #
    # for layer in model.mask_model.encoder.parameters():
    #     layer.requires_grad = False
    # log_msg(f'UPDATE | frozen encoder weights')
    # ---- MAP INFERENCE DECODER WEIGHTS ---- #
    # for k in pretrain_keys[1:]:
    #     if 'decoder_reconstruction' in k:
    #         new_key = k.replace('decoder_reconstruction', 'decoder_inference')
    #         model.state_dict()[new_key] = model.state_dict()[new_key].copy_(pretrain_dict[k])
    #         log_msg(f'UPDATE | mapped: {k} to {new_key}')




best_loss = 1e30
best_acc = 0
best_lk = 1e30
global_step = 0

training_losses = []
validation_losses = []

train_dice = []
test_dice = []
train_dice_iqr = []
test_dice_iqr = []
inference_dice = []

dims = train_lesions.shape[2:]

# --- set epochs to account for changes in training set --- #

# model_params['EPOCHS'] = int(model_params['EPOCHS'] )

inference_predictions = np.zeros([model_params['EPOCHS'], *dims])    
# ---- prepare training sets ---- #

# set_order = []
# for i in range(len(deficits)):
#     tmp = list(np.repeat(i,repetition_factor))
#     set_order.append(tmp)

# set_order = list(np.array(set_order).reshape(-1))
# # training_index = list(np.array(np.array([np.repeat(0, 10), np.repeat(1, 10), np.repeat(2, 10)])).reshape(-1))
# training_index = set_order * dataset_reps

# # training_index = np.random.permutation(training_index).tolist()
# # int(int(model_params['EPOCHS'] // len(training_sets))/repetition_factor)




for epoch in range(model_params['EPOCHS']):
    # training_set = deficits[training_index[epoch]]
    training_set = deficits[0]
    model.zero_grad()
    train_acc = 0
    t_epoch_loss = 0
    batch_dice = []
    # The trackers for the mean and scale of the inference map
    vae_mask = np.zeros((dims))
    vae_scale = np.zeros((dims))
    for (x, y) in train_loaders:
        optimizer.zero_grad()
        x = x.type(Tensor).to(device)
        y = y.type(Tensor).to(device)
        ret_dict = model(x, y)
        # ---- adjust loss weighting ---- #
        # loss = ret_dict['loss'].mean()
        loss = ret_dict['mask_ll'] + ret_dict['recon_ll'] + ret_dict['kl']
        # if epoch > 20:
        #     loss = ret_dict['mask_ll']  + 0*ret_dict['recon_ll'] + 0.01*ret_dict['kl']
        #     for layer in model.mask_model.decoder_reconstruction.parameters():
        #         layer.requires_grad = False
        # else:
        #     loss = ret_dict['mask_ll'] + ret_dict['recon_ll'] + ret_dict['kl']
        # # if epoch == 5:
        #     initial_loss = ret_dict['loss'].mean()
        # loss = ret_dict['mask_ll'] + (1-(epoch/model_params['EPOCHS'])) * ret_dict['recon_ll'] + ret_dict['kl']
        # ---- unfreeze reconstruction decoder after 20 epochs ---- #
        # if epoch > 20:
        #     for layer in model.mask_model.decoder_reconstruction.parameters():
        #         layer.requires_grad = True
        t_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        vae_mask += np.squeeze(ret_dict['mean_mask'].cpu().data.numpy())
        vae_scale += np.squeeze(ret_dict['mask_scale'].cpu().data.numpy())
        train_acc += 1
        global_step += 1
        pred = ret_dict['lesion_recon'].detach().cpu().numpy()
        for l in range(pred.shape[0]):
            pred[l] = np.where(pred[l]>np.quantile(pred[l],0.95),1,0)
        target = x.cpu().data.numpy()
        batch_dice.append(vec_dice(pred[:,0,:,:,:], target[:,0,:,:,:]))
    train_dice.append(np.mean(np.concatenate(batch_dice)))
    train_dice_iqr.append(scipy.stats.iqr(np.concatenate(batch_dice)))
    training_losses.append(t_epoch_loss / train_acc)
    vae_mask = vae_mask / train_acc
    val_mask = torch.from_numpy(vae_mask).type(Tensor).to(device).view(1, 1,*dims)
    vae_scale = vae_scale / train_acc
    val_scale = torch.from_numpy(vae_scale).type(Tensor).to(device).view(1, 1,*dims)
    val_acc = 0
    accuracy_acc = 0
    loss_acc = 0
    likelihood_acc = 0
    kld_acc = 0
    recon_acc = 0
    batch_dice = []
    with torch.no_grad():
        for (x, y) in test_loader:
            x = x.type(Tensor).to(device)
            y = y.type(Tensor).to(device)
            ret_dict = model(x, y,
                             provided_mask=val_mask,
                             provided_scale=val_scale,
                             val=True)
            loss_acc += ret_dict['loss'].mean().item()
            val_acc += 1
            likelihood_acc += ret_dict['mask_ll'].item()
            accuracy_acc += ret_dict['acc'].item()
            kld_acc += ret_dict['kl'].item()
            recon_acc += ret_dict['recon_ll'].item()
            pred = ret_dict['lesion_recon'].detach().cpu().numpy()
            for l in range(pred.shape[0]):
                pred[l] = np.where(pred[l]>np.quantile(pred[l],0.95),1,0)
            target = x.cpu().data.numpy()
            batch_dice.append(vec_dice(pred[:,0,:,:,:], target[:,0,:,:,:]))
    test_dice.append(np.mean(np.concatenate(batch_dice)))
    test_dice_iqr.append(scipy.stats.iqr(np.concatenate(batch_dice)))
    loss = loss_acc / val_acc
    validation_losses.append(loss)
    lk = likelihood_acc / val_acc
    acc = round(accuracy_acc / val_acc, 4)
    kl = round(kld_acc / val_acc, 3)
    rec = recon_acc / val_acc
    inference_predict = ret_dict['mean_mask'].cpu().data.numpy().reshape(dims)
    inference_predictions[epoch,:,:] = inference_predict
    pred = np.where(inference_predict>np.quantile(inference_predict,0.95),1,0)
    inference_dice.append(dice_3D(pred, substrate))
    # print(f'Epoch: {epoch}, mask likelihood: {lk}, KL: {kl}, recon likelihood: {rec}')
    if lk < best_lk:
        best_loss = loss
        best_lk = lk
        best_acc = acc
        best_recon = recon_acc
        best_epoch = epoch
        torch.save(model, f"vae.pth")
        np.save(f'vae_mask.npy', vae_mask)
        np.save(f'vae_scale.npy', vae_scale)
    # if epoch % 10 == 0:
        # log_msg(f'UPDATE | Best: {best_lk}, {best_loss}, {best_acc}, epoch: {best_epoch}')
    # VIZUALISE AS THE TRAINING GOES ON
    if epoch % 20 == 0:
        imgs = x.cpu().data.numpy()
        recons = ret_dict['lesion_recon'].cpu().data.numpy()
        # inference_predict = ret_dict['mean_mask'].cpu().data.numpy().reshape(32,32)
        visualize_inference3D(substrate, inference_predict, template_brain, os.path.join(out_dir, f'inference_epoch_{epoch}.png'))
        visualize_inference3D(imgs[0,:,:].reshape(dims), recons[0,:,:].reshape(dims), template_brain, os.path.join(out_dir, f'reconstruction_epoch_{epoch}.png'))
        log_msg(f'UPDATE | loss: {training_losses[epoch]}, train-dice: {train_dice[epoch]}, epoch: {epoch}')


#########################################
#                                       #
#             SAVING RESULTS            #
#                                       #
#########################################

np.save(os.path.join(out_dir, f'inference_predictions.npy'), inference_predictions)
np.save(os.path.join(out_dir, f'dice_training.npy'), train_dice)
np.save(os.path.join(out_dir, f'dice_validation.npy'), test_dice)
np.save(os.path.join(out_dir, f'dice_inference.npy'), inference_dice)



#########################################
#                                       #
#           VISUALISE RESULTS           #
#                                       #
#########################################

# ---- substrate predictions ---- #
for th in [0.25,0.5,0.75,0.9,0.95, 0.99, 0.995]:
    tmp = inference_predictions[epoch,:,:,:]  # Updated for 3D
    testing = np.where(tmp>np.quantile(tmp,th),1,0)
    visualize_inference3D(substrate, testing, template_brain, os.path.join(out_dir, f'Inference_threshold_{th}.png'))


# ---- training / testing performance ---- #
plt.plot(train_dice, label = 'recon | training dice')
plt.fill_between(np.arange(len(train_dice)), train_dice - np.array(train_dice_iqr), train_dice + np.array(train_dice_iqr), alpha=0.2)
plt.plot(test_dice, label = 'recon | validation dice')
plt.fill_between(np.arange(len(test_dice)), test_dice - np.array(test_dice_iqr), test_dice + np.array(test_dice_iqr), alpha=0.2)
plt.plot(inference_dice, label = 'inference | validation dice')
plt.legend()
plt.title(f'Mean dice scores | {test_lesions.shape[0]} lesions')
plt.xlabel('epoch')
plt.ylabel('dice coefficient')
plt.savefig(out_dir + '/dice_scores.png')
plt.close()


shutil.make_archive(os.path.join('/data', f'{out_dir}'), 'zip', out_dir)



log_msg("FINISHED | running deep lesion deficit mapping")
