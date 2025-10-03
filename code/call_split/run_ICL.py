#
#
#
# This script runs the initial in-context learning 
# deep lesion deficit mapping experiments
#
# The initial set up focuses on extracting complex lesion-deficit behaviour relationships
# The approach entails:
# 1. pretraining the model using simple behaviour relationships (lesion overlap, size-ratio)
# 2. fine-tuning the model using complex behaviour relationships (center of gravity distance)


import scipy.io, os, json
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim

os.chdir('call_split')
import torch as tc


# from model import *
from model import *
from utils import log_msg, get_variable, get_device, DeficitDataset, visualize_inference2D, count_parameters, vec_dice, dice_2D, get_deficit

log_msg("START | running deep lesion deficit mapping")

#########################################
#                                       #
#              PARSE INPUT              #
#                                       #
#########################################

log_msg("UPDATE | parsing input variables")
# ---- template path ---- #
Path = get_variable('TEMPLATEDIR')

# ---- lesion subset count ---- #
n_lesions = get_variable('N_LESIONS')
# set N for initial testing purposes
# n_lesions = 1000
if n_lesions:
    n_lesions = int(n_lesions)

# ---- output directory ---- #

out_dir = get_variable('OUTDIR')
if not out_dir:
    out_dir = '/data/out_dir'
else:
    out_dir = os.path.join('/data', out_dir)

if not os.path.isdir(out_dir):
    os.makedirs(out_dir, exist_ok=True)

log_msg(f'UPDATE | output directory: {out_dir}')

# ---- pretraining  ---- #
pretraining = get_variable('PRETRAINING')
if pretraining:
    pretraining = eval(pretraining)
    if pretraining:
        log_msg(f'UPDATE | utilizing pretrained model')

# ---- pretraining model weights ---- #
model_path = get_variable('MODEL_PATH')
if not model_path: 
    model_path = os.path.join('/data','pretrain_20K','recon_vae.pth')

if pretraining:
    log_msg(f'UPDATE | using pretrained model weights: {model_path}')

# ---- anatomically constrained inference  ---- #
aci = get_variable('ACI')
if aci:
    aci = eval(aci)
    if aci:
        log_msg(f'UPDATE | utilizing anatomically constrained model')



# ---- lesion set ---- #

lesion_type = get_variable('LESION_TYPE')

if not lesion_type:
    lesion_type = '5000_lesions_2D.npy'

# lesion_type = 'icl_20K_2D.npy'

log_msg(f'UPDATE | lesion type: {lesion_type}')


# ---- substrate ---- #

# first substrate
# substrate_type = 'cognition_substrate_2D.npy'
# second substrate
# substrate_type = 'motor_substrate_2D.npy'
substrate_type = get_variable('SUBSTRATE_TYPE')

if not substrate_type:
    substrate_type = 'two_point_substrate_2D.npy'

substrate = np.load(os.path.join(Path,'validation',substrate_type))

# ---- multiple NQ based substrates ---- #

# Networks=['Hearing','Language','Introspection','Cognition','Mood','Memory','Aversion','Coordination','Interoception','Sleep','Reward','Visual','Spatial','Somatosensory']

# # use for validation:'Motor'
# test_substrate = 'Motor'
# log_msg(f'UPDATE | using multiple NeuroQuery based substrates')
# substrates = dict()
# for n in Networks:
#     substrates[n] = np.load(os.path.join('/data','substrates/maps', f'Giles_et_al_2013_{n}_2D.npy'))
#     log_msg(f'UPDATE | using substrate: {n}')

# ---- single NQ based substrates ---- #
# Networks=['Motor']

# use for validation:'Motor'
# test_substrate = 'Motor'
# log_msg(f'UPDATE | using multiple NeuroQuery based substrates')
# substrates = dict()
# for n in Networks:
#     substrates[n] = np.load(os.path.join('/data','substrates/maps', f'Giles_et_al_2013_{n}_2D.npy'))
#     log_msg(f'UPDATE | using substrate: {n}')
test_substrate = substrate_type
# ---- deficit scores ---- #

deficits_train = ['trans']
# deficits_train = ['overlap_binary','overlap_ratio_noisy']
# deficits_train = ['distance']
# deficits_train = ['overlap_binary','size', 'distance']

# deficits_train = ['overlap_binary','overlap_ratio_noisy']
# deficits_test = ['distance']
deficits_test = ['trans']

# deficit_type = get_variable('DEFICIT_TYPE')

# if not deficit_type:
#     deficit_type = 'overlap_ratio_noisy'

# log_msg(f'UPDATE | deficit type: {deficit_type}')



#########################################
#                                       #
#               DATA I/O                #
#                                       #
#########################################

# ---- lesions ---- #
# train_lesions = np.load(os.path.join('/data','pretrain',lesion_type))
train_lesions = np.load(os.path.join(Path,'validation',lesion_type))


# test_lesions = np.load(os.path.join('/data','pretrain','validation_10K_2D.npy'))
test_lesions = np.load(os.path.join(Path,'validation',lesion_type))

test_lesions = test_lesions[4000:,:,:]

# ---- ensure non-empty lesions ---- #

sum_check = np.sum(train_lesions, axis=(1,2))
empty_lesion = np.where(sum_check==0)
train_lesions = np.delete(train_lesions, empty_lesion, axis=0)
log_msg(f'UPDATE | number of empty training lesions removed: {len(empty_lesion[0])}')

sum_check = np.sum(test_lesions, axis=(1,2))
empty_lesion = np.where(sum_check==0)
test_lesions = np.delete(test_lesions, empty_lesion, axis=0)
log_msg(f'UPDATE | number of empty validation lesions removed: {len(empty_lesion[0])}')


# ---- select consistent lesion subset ---- #
# first set of lesions for inference pre-training
n_lesions = 2500
if n_lesions:
    train_lesions = train_lesions[:n_lesions,:,:]
    log_msg(f'UPDATE | using random {n_lesions} lesions')

log_msg(f'UPDATE | number of training lesions: {train_lesions.shape[0]}')
log_msg(f'UPDATE | number of validation lesions: {test_lesions.shape[0]}')

# ---- load template brain ---- #
template_brain = np.load(os.path.join(Path,'validation','MNI152_T1_32.npy'))
# template_brain = np.rot90(np.load(os.path.join(Path,'validation','mni_brain_32.npy'))[16,:,:],1)
template_brain = np.rot90(np.sum(np.load(os.path.join(Path,'validation','mni_brain_32.npy')), axis = 0),1)

# ---- calculate deficit scores ---- #
scores_train = dict()
# noise_levels = [0.1, 0.5, 1.0]
noise_levels = [0.25]


# for net in Networks:
#     substrate = substrates[net]
#     log_msg(f'UPDATE | calculating deficits for substrate: {net}')
#     for deficit_type in deficits_train:
#         for n in noise_levels:
#             if deficit_type == 'overlap_ratio_noisy':
#                 idx = f'{net}-{deficit_type}-{n}'
#                 scores_train[idx] = get_deficit(train_lesions, substrate, deficit_type, n)
#             else:
#                 idx = f'{net}-{deficit_type}'
#                 scores_train[idx] = get_deficit(train_lesions, substrate, deficit_type, 0.25)

scores_train['trans'] = get_deficit(train_lesions,substrate, 'trans', 0.25)

deficits = list(scores_train.keys())


test_substrate = np.load(os.path.join('/data','substrates/maps', f'Giles_et_al_2013_{test_substrate}_2D.npy'))

scores_test = get_deficit(test_lesions, test_substrate, deficits_test[0], 0.25)



# ---- VISUALIZATIONS ---- #
for net in Networks:
    substrate = substrates[net]
    log_msg(f'UPDATE | visualizing substrate: {net}')
    visualize_inference2D(substrate, substrate, template_brain, out_dir + f'/{net}_overlay.png')

aggregate = np.sum(train_lesions, axis=0)
visualize_inference2D(aggregate, aggregate, template_brain, out_dir + '/train_lesions_aggregate.png')

aggregate = np.sum(test_lesions, axis=0)
visualize_inference2D(aggregate, aggregate, template_brain, out_dir + '/test_lesions_aggregate.png')

fig = plt.figure(figsize=(25., 25.))
grid = ImageGrid(fig, 111, 
                 nrows_ncols=(2, 5),
                 axes_pad=0.05,
                 share_all=True
                 )

for i in range(10):
    grid[i].imshow(train_lesions[i])

plt.savefig(out_dir + '/10_train_lesions.png')
plt.close()

for deficit_type in deficits:
    plt.hist(scores_train[deficit_type])
    plt.title(f'histogram of {deficit_type} deficit')
    plt.savefig(out_dir + f'/{deficit_type}_histogram.png')
    plt.close()

plt.hist(scores_test)
plt.title(f'histogram of validation deficit')
plt.savefig(out_dir + f'/validation_histogram.png')
plt.close()

#########################################
#                                       #
#           MODEL PARAMETERS            #
#                                       #
#########################################

params = ['INPUT_SIZE','CONTINUOUS', 'Z_DIM','EPOCHS','INITIAL_CONV_KERNELS','L2_REG','LR', 'LATENT_SPLIT']


# ---- load default parameters ---- #
with open(os.path.join(Path,'validation','model_param_defaults.json')) as f:
    model_params = json.load(f)
    print(model_params)

# ---- update with user parameters ---- #
for p in params:
    locals()[p] = get_variable(p)
    if locals()[p]:
        model_params[p] = eval(locals()[p])

# ---- save final model parameters ---- #
with open(os.path.join(out_dir,'model_parameters.json'), "w") as f:
        json.dump(model_params, f, indent=4)

model_params['INPUT_SIZE'] = train_lesions.shape[-1]

log_msg('UPDATE | using model parameters:')
for p in model_params.keys():
    log_msg(f'UPDATE | {p}: {model_params[p]}')


#########################################
#                                       #
#                DATASETS               #
#                                       #
#########################################

# ---- add color channel ---- #
if len(train_lesions.shape) < 4:
    train_lesions = np.expand_dims(train_lesions, axis=1)

if len(test_lesions.shape) < 4:
    test_lesions = np.expand_dims(test_lesions, axis=1)

# ---- determine batch size ---- #
n_samples = train_lesions.shape[0]

if n_samples > 511:
    batch_size = 256
else:
    batch_size = int(np.round(n_samples // 10,0))

log_msg(f'UPDATE | batch size: {batch_size}')

# ---- prepare training dataset using various deficit scores ---- #


datasets = dict()

for deficit_type in deficits:
    datasets[deficit_type] = DeficitDataset(data=train_lesions, labels=scores_train[deficit_type])


# dataset = DeficitDataset(data=train_data, labels=train_labels)
train_loaders = dict()
for deficit_type in deficits:
    train_loaders[deficit_type] = DataLoader(datasets[deficit_type], batch_size=batch_size, drop_last=False,shuffle=True, num_workers=0, pin_memory=True)

# ---- prepare testing dataset using complex deficit scores ---- #

# VALIDATION
val_dataset = DeficitDataset(data=test_lesions, labels=scores_test)
val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False,
                                            shuffle=True, num_workers=0, pin_memory=True)


#################################
# device = torch.device("cuda:0")
device = get_device()



Tensor = torch.cuda.FloatTensor

torch.manual_seed(42)

model = ModelWrapper(model_params['INPUT_SIZE'],
                     z_dim=model_params['Z_DIM'],
                     start_dims=model_params['INITIAL_CONV_KERNELS'],
                     continuous=model_params['CONTINUOUS'],
                     aci=aci,
                     template = np.where(template_brain>0,1,0),
                     latent_split=model_params['LATENT_SPLIT']).to(device)

# Other optimisers work as well, Adamax is quite stable though
optimizer = optim.Adamax(model.parameters(),
                         weight_decay=model_params['L2_REG'],
                         lr=model_params['LR'])

log_msg('UPDATE | model paramter count: {}'.format(count_parameters(model)))

# ---- set epochs to account for changes in training set ---- #
repetition_factor = 5
dataset_reps = 5
model_params['EPOCHS'] = len(deficits) * repetition_factor * dataset_reps
p = 'EPOCHS'
log_msg(f'UPDATE | {p}: {model_params[p]}')


#################################
#                               #
#        USE PRETRAINING        #
#                               #
#################################

if pretraining:
    # ---- load pretrained weights ---- #
    pretrain_dict = torch.load(model_path, map_location=device)
    pretrain_keys = list(pretrain_dict.keys())
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

# model_params['EPOCHS'] = 210

inference_predictions = np.zeros([model_params['EPOCHS'], *dims])    
# ---- prepare training sets ---- #

set_order = []
for i in range(len(deficits)):
    tmp = list(np.repeat(i,repetition_factor))
    set_order.append(tmp)

set_order = list(np.array(set_order).reshape(-1))
# training_index = list(np.array(np.array([np.repeat(0, 10), np.repeat(1, 10), np.repeat(2, 10)])).reshape(-1))
training_index = set_order * dataset_reps

# training_index = np.random.permutation(training_index).tolist()
# int(int(model_params['EPOCHS'] // len(training_sets))/repetition_factor)




for epoch in range(model_params['EPOCHS']):
    training_set = deficits[training_index[epoch]]
    model.zero_grad()
    train_acc = 0
    t_epoch_loss = 0
    batch_dice = []
    # The trackers for the mean and scale of the inference map
    vae_mask = np.zeros((dims))
    vae_scale = np.zeros((dims))
    for (x, y) in train_loaders[training_set]:
        optimizer.zero_grad()
        x = x.type(Tensor).to(device)
        y = y.type(Tensor).to(device)
        ret_dict = model(x, y)
        loss = ret_dict['loss'].mean()
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
        batch_dice.append(vec_dice(pred, target))
    train_dice.append(np.mean(np.concatenate(batch_dice)))
    train_dice_iqr.append(scipy.stats.iqr(np.concatenate(batch_dice)))
    training_losses.append(t_epoch_loss / train_acc)
    vae_mask = vae_mask / train_acc
    val_mask = tc.from_numpy(vae_mask).type(Tensor).to(device).view(1, 1,*dims)
                                                                #    model_params['INPUT_SIZE'],
                                                                #    model_params['INPUT_SIZE'])
    vae_scale = vae_scale / train_acc
    val_scale = tc.from_numpy(vae_scale).type(Tensor).to(device).view(1, 1,*dims)
                                                                #    model_params['INPUT_SIZE'],
                                                                #    model_params['INPUT_SIZE'])
    val_acc = 0
    accuracy_acc = 0
    loss_acc = 0
    likelihood_acc = 0
    kld_acc = 0
    recon_acc = 0
    batch_dice = []
    with torch.no_grad():
        for (x, y) in val_loader:
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
            batch_dice.append(vec_dice(pred, target))
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
    inference_dice.append(dice_2D(pred, test_substrate))
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
        visualize_inference2D(test_substrate, inference_predict, template_brain, os.path.join(out_dir, f'inference_epoch_{epoch}.png'))
        visualize_inference2D(imgs[0,:,:].reshape(dims), recons[0,:,:].reshape(dims), template_brain, os.path.join(out_dir, f'reconstruction_epoch_{epoch}.png'))
        log_msg(f'UPDATE | loss: {loss}, train-dice: {train_dice[epoch]}, epoch: {epoch}')


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
for th in [0.25,0.5,0.75,0.9,0.95]:
    tmp = inference_predictions[epoch,:,:]
    testing = np.where(tmp>np.quantile(tmp,th),1,0)
    visualize_inference2D(test_substrate, testing, template_brain, os.path.join(out_dir, f'Inference_threshold_{th}.png'))


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



