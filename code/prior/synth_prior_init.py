#########################################################################
#                                                                       #
# This script contains code for the initial generation of               #
# synthetic 3D lesion masks for Ischaemic stroke lesions.               #
#                                                                       #
#                                                                       #
# Author: Patrik Bey, patrik.bey@ucl.ac.uk                              #
#                                                                       #
# last update: 2025/11/02                                               #
#                                                                       #
#                                                                       #
#########################################################################



# ---- load libraries ---- #
import scipy.io, os, json, numpy as np, matplotlib.pyplot as plt, nibabel, torch, torch.optim as optim, sklearn, shutil, progress.bar

from skimage import measure
from torch.utils.data import Dataset, DataLoader

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from model import *
from utils import *
# log_msg, get_variable, get_device, LesionDataset, visualize_inference3D, count_parameters, vec_dice, get_deficit


device = get_device()

#########################################
#                                       #
#              LOAD INPUT               #
#                                       #
#########################################

Path = '/mnt/h/DLDM/3D'
MODEL_PATH=os.path.join(Path, 'pretrain-recon', 'recon_vae.pth')

OUT_DIR = os.path.join(Path, 'lesions')

mni_brain = nibabel.load(os.path.join(Path,'templates','MNI152_64_brain.nii.gz')).get_fdata()
mni_brain_mask = np.where(mni_brain>0,1,0)



##################################
#                                #
#           BUILD MODEL          #
#                                #
##################################

# ---- set functions ---- #
Tensor = torch.cuda.FloatTensor


# ---- load pretrained model ---- #
pretrain_dict = torch.load(MODEL_PATH, map_location=device)

pretrain_keys = list(pretrain_dict.keys())

# ---- extract model parameters ---- #
INITIAL_CONV_KERNELS = pretrain_dict[pretrain_keys[0]].shape[0]
Z_DIM = pretrain_dict['mask_model.mu.weight'].shape[0]
CONTINUOUS = False
INPUT_SIZE = mni_brain.shape[0]
DIMS = mni_brain.shape


# ---- build model ---- #
model = ModelWrapperRecon(INPUT_SIZE,
                     z_dim=Z_DIM,
                     start_dims=INITIAL_CONV_KERNELS,
                     continuous=CONTINUOUS,
                     in_channels=1, # only lesion mask input channel
                     lesion_threshold=False).to(device)


# ---- load weights into model ---- #
model.load_state_dict(pretrain_dict)
log_msg('UPDATE | model parameter count: {}'.format(count_parameters(model)))





##################################
#                                #
#          CREATE PRIORS         #
#                                #
##################################

# N = 500
# RUNS = 40
# with progress.bar.Bar('Processing', max=N*RUNS) as bar:
#     for run in range(RUNS):
#         initial_masks = sample_latent_masks(model, num_samples=N)
#         masks = np.zeros(initial_masks.shape)
#         for i in range(N):
#             tmp = initial_masks[i].reshape(DIMS).cpu().data.numpy().astype(int)
#             tmp = np.where(tmp>=np.quantile(tmp,0.995),1,0)
#             tmp = tmp * mni_brain_mask
#             img = measure.label(tmp, background=0)
#             props = measure.regionprops(img)
#             if len(props)>1:
#                 chance = np.random.rand()
#                 if chance>0.5:
#                     clusters = np.random.randint(0,len(props))
#                     mask = np.where(np.logical_and(img>0,img<=clusters+1),1,0 )
#             else:
#                 mask = tmp
#             masks[i,0,:,:,:] = mask
#             bar.next()
#         np.save(os.path.join(OUT_DIR, f'Synthetic_lesion_masks_3D_{run}.npy'), masks.astype(np.int32))



# for i in [10,20,30,40,49]:
#     tmp = masks[i,0,:,:,:]
#     nibabel.save(nibabel.Nifti1Image(tmp.astype(np.float32), np.eye(4)), os.path.join(OUT_DIR, f'synthetic_lesion_{i}.nii.gz'))


RUNS = 40
lesions = np.zeros([20000,1, 64,64,64], dtype=np.int32)

with progress.bar.Bar('Processing', max=RUNS) as bar:
    for run in range(RUNS):
        tmp = np.load(os.path.join(OUT_DIR, f'Synthetic_lesion_masks_3D_{run}.npy'), allow_pickle=True)
        lesions[run*500:(run+1)*500,:,:,:,:] = tmp
        bar.next()


np.save(os.path.join(OUT_DIR, f'Synthetic_lesions_3D.npy'), lesions)


##################################
#                                #
#          VALIDATE MASKS        #
#                                #
##################################

lesions = np.load(os.path.join(OUT_DIR, 'Ischaemic_lesions_3D.npy')).astype(np.int32)

metrics = ['area','equivalent_diameter_area', 'axis_major_length', 'axis_minor_length', 'euler_number','extent']

# ---- MEASURES FOR REAL LESIONS ---- #
MEASURES = np.zeros([lesions.shape[0], len(metrics)+1])
with progress.bar.Bar('Processing', max=lesions.shape[0]) as bar:
    for i in range(lesions.shape[0]):
        tmp = lesions[i,:,:,:]
        try:
            props = measure.regionprops_table(tmp,properties=metrics)
            props = np.array([props[m][0] for m in metrics])
            MEASURES[i, :6] = props
            img=measure.label(tmp, background=0)
            props = measure.regionprops(img)
            MEASURES[i, 6] = len(props)
        except:
            MEASURES[i, :] = np.zeros(len(metrics)+1)
        bar.next()


# ---- clean measures ---- #
check = np.sum(MEASURES, axis=1)
MEASURES_clean = MEASURES[check>0,:]

np.save(os.path.join(OUT_DIR, 'Ischaemic_lesion_measures.npy'), MEASURES_clean)


# ---- MEASURES FOR SYNTHETIC LESIONS ---- #


MEASURES = np.zeros([lesions.shape[0], len(metrics)+1])

with progress.bar.Bar('Processing', max=lesions.shape[0]) as bar:
    for i in range(lesions.shape[0]):
        tmp = lesions[i,:,:,:].reshape(DIMS)
        try:
            props = measure.regionprops_table(tmp,properties=metrics)
            props = np.array([props[m][0] for m in metrics])
            MEASURES[i, :6] = props
            img=measure.label(tmp, background=0)
            props = measure.regionprops(img)
            MEASURES[i, 6] = len(props)
        except:
            MEASURES[i, :] = np.zeros(len(metrics)+1)
        bar.next()

# ---- clean measures ---- #
check = np.sum(MEASURES, axis=1)
MEASURES_clean = MEASURES[check>0,:]


np.save(os.path.join(OUT_DIR, 'Synthetic_lesion_measures.npy'), MEASURES_clean)



# ---- t-SNE MAPPING ---- #

metrics = ['area','equivalent_diameter_area', 'axis_major_length', 'axis_minor_length', 'euler_number','extent', 'cluster count']


real_measures = np.load(os.path.join(OUT_DIR, 'Ischaemic_lesion_measures.npy'))
synth_measures = np.load(os.path.join(OUT_DIR, 'Synthetic_lesion_measures.npy'))
all_measures = np.concatenate([real_measures, synth_measures], axis=0)


x = StandardScaler().fit_transform(all_measures)


# for m in range(7):
#     plt.subplot(2,4,m+1)
#     plt.boxplot(all_measures[:4944,m], positions=[1], widths=0.6)
#     plt.boxplot(all_measures[4944:,m], positions=[2], widths=0.6)
#     plt.xticks([1,2], ['Real','Synthetic'])
#     plt.title(metrics[m])


tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
tsne_results = tsne.fit_transform(x)


X_embedded_min = np.min(tsne_results, axis=0)
X_embedded_max = np.max(tsne_results, axis=0)
X_embedded = 200 * (tsne_results - X_embedded_min) / (X_embedded_max - X_embedded_min)



real_tsne = X_embedded[:4944,:]
synth_tsne = X_embedded[4944:,:]

real_mapping = np.zeros([201,201])
for i in range(real_tsne.shape[0]):
    tmp = real_tsne[i,:]
    real_mapping[tmp[0].astype(int), tmp[1].astype(int)] += 1 


# real_smooth = ndimage.gaussian_filter(real_mapping, sigma=2)

# plt.contourf(np.where(real_smooth.T>0.05, real_smooth.T, np.nan), cmap = 'Reds', filled=True, alpha=0.75)

# plt.scatter(synth_tsne[:, 0]-np.min(synth_tsne[:,0]), synth_tsne[:, 1]-np.min(synth_tsne[:,1]), marker='h', color='Blue', edgecolors='white',linewidths=0.5, alpha = .25)

plt.figure(figsize=(25,25))
count=1
for i in range(7):
    for j in range(7):
        plt.subplot(7,7,count)
        plt.scatter(synth_measures[:5000, i], synth_measures[:5000, j], marker='h', color='mediumvioletred',linewidths=0.5, alpha = .1, label = 'Synthetic lesions')
        plt.scatter(real_measures[:, i], real_measures[:, j], marker='h', color='gold',linewidths=0.5, alpha = .1, label = 'Real lesions')
        count += 1
        plt.xlabel(metrics[i])
        plt.ylabel(metrics[j])
        plt.title(f'{metrics[i]} vs {metrics[j]}')

plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'measure_scatterplots.png'))
plt.close()



for i in range(7):
    plt.subplot(2,4,i+1)
    plt.hist(synth_measures[:, i], bins=30, color='mediumvioletred', alpha=0.5, label='Synthetic lesions')
    plt.hist(real_measures[:, i], bins=30, color='gold', alpha=0.5, label='Real lesions')
    plt.xlabel(metrics[i])
    plt.ylabel('Count')

plt.legend()

plt.show()