#
#
# data_creation.py
#
# authors :
# 2. Bey, Patrik
#
#
#
#
# This script contains snippets to
# create the data for the model.
#
# Data created within this script:
#
# 50K artificial lesion masks split
# into three distinct sets for both 2D and 3D
#
# 1. recon_20K | pretraining dataset used for reconstruction pretraining
# 2. icl_20K | main training dataset used for ICL pretraining
# 3. validation_10K | validation dataset for few-shot training / validation
######

###################################
#                                 #
#         LOAD LIBRARIES          #
#                                 #
###################################


import os, numpy, nibabel, matplotlib.pyplot as plt, scipy.ndimage, progress.bar
###################################
#                                 #
#         FUNCTIONS               #
#                                 #
###################################

# def get_overlay_ratio(img, mask):
#     '''
#     return the ratio of the substrate mask	 
#     and the lesion mask
#     '''
#     overlay = array1*array2


# def get_deficit_score(lesions, substrate, binary=True):
#     '''
#     return the deficit score for a given substrate and lesion mask
#     defined as binary value if the ratio of the lesion mask to the substrate mask is greater than 5%
#     '''
#     deficit_scores = [0 for i in range(len(lesions))]
#     for i in range(len(lesions)):
#         overlap = lesions[i] * substrate
#         counts = numpy.count_nonzero(overlap)
#         voxels_gt = numpy.sum(substrate)
#         ratio_lesion = counts / voxels_gt
#         if binary:
#             if ratio_lesion > 0.05:
#                 deficit_scores[i] = 1 
#             else:
#                 deficit_scores[i] = 0
#         else:
#             deficit_scores[i] = ratio_lesion
#     return(deficit_scores)

# def vec_dice(array1,array2):
#     '''
#     compute dice scores between two 4D arrays
#     input:
#         4D arrays of lesion masks
#     output:
#         1D vector of dice scores of all masks
#     '''
#     import numpy
#     im1 = numpy.asarray(array1).astype(bool)
#     im2 = numpy.asarray(array2).astype(bool)
#     if im1.shape != im2.shape:
#         raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
#     intersection = numpy.logical_and(im1, im2)
#     return(2. * intersection.sum(axis=tuple(range(1,4))) / (im1.sum(axis=tuple(range(1,4))) + im2.sum(axis=tuple(range(1,4)))))


# array1=lesions.copy()
# array2=img_vec.copy()

###################################
#                                 #
#      CREATING 3D SUBSTRATES     #
#                                 #
###################################
'''

# ---- 0. LOAD DATA ---- #
mni = nibabel.load(os.path.join('/data/MNI152_T1_1mm.nii.gz'))
mni_img = mni.get_fdata()
mni_affine = mni.affine

quest = numpy.load(os.path.join(os.getenv('TEMPLATEDIR'), 'toy_examples/question_mark_substrate.npy'))
# zet = numpy.load(os.path.join(os.getenv('TEMPLATEDIR'), 'toy_examples/double_dot_substrate.npy'))


# ---- 1. 3D question mask substrate ---- #

img = numpy.zeros([quest.shape[0], quest.shape[1], quest.shape[0]])

for i in range(6):
    img[13+i,:,:] = numpy.flip(numpy.rot90(quest,1),0)

nibabel.save(nibabel.Nifti1Image(numpy.flip(img, axis=2), numpy.eye(4)), os.path.join('/data', 'question_mark_substrate_3D.nii.gz'))


# ---- 2. 3D two point substrate ---- #

img = numpy.zeros([182,218,182])

# create center of blobs
img[55,75,55] = 1
img[141,141,141] = 1


smimage = scipy.ndimage.gaussian_filter(img, float(2))
bin_img = numpy.where(smimage > numpy.quantile(smimage, 0.95), 1, 0).astype(float)


# READJUST TWO DOT LESION MASKS SIZES


res = resize(bin_img, (32,32,32))
nibabel.save(nibabel.Nifti1Image(numpy.flip(res, axis=2),numpy.eye(4)), os.path.join('/data', 'two_point_substrate_3D_bin_32.nii.gz'))



# nibabel.save(nibabel.Nifti1Image(numpy.flip(smimage, axis=2), numpy.eye(4)), os.path.join('/data', 'two_point_substrate_3D.nii.gz'))

nibabel.save(nibabel.Nifti1Image(numpy.flip(bin_img, axis=2), mni.affine), os.path.join('/data', 'two_point_substrate_3D_bin.nii.gz'))


# smimg = scipy.ndimage.gaussian_filter(numpy.flip(bin_img, axis=2), float(1))
# smimg = numpy.where(smimg > numpy.quantile(smimg, 0.975), 1, 0).astype(float)
# nibabel.save(nibabel.Nifti1Image(smimg, numpy.eye(4)), os.path.join('/data', 'question_mark_substrate_3D_sm.nii.gz'))

# smimg = scipy.ndimage.gaussian_filter(numpy.flip(bin_img, axis=2), float(1))
# smimg = numpy.where(smimg > numpy.quantile(smimg, 0.975), 1, 0).astype(float)
# nibabel.save(nibabel.Nifti1Image(smimg, numpy.eye(4)), os.path.join('/data', 'question_mark_substrate_3D_sm.nii.gz'))

# ---- 3. 3D lesion masks ---- #

lesions = numpy.load(os.path.join(os.getenv('TEMPLATEDIR'),'toy_examples','1000_lesions.npy'))

masks = numpy.zeros([lesions.shape[0], lesions.shape[1], lesions.shape[1], lesions.shape[1]])

for i in range(lesions.shape[0]):
    idx = numpy.random.randint(0, lesions.shape[1])
    masks[i,:,idx,:] = lesions[i,:,:]

smmask = numpy.zeros([masks.shape[0], masks.shape[1], masks.shape[2], masks.shape[3]])

for i in range(masks.shape[0]):
    sm =  abs(numpy.random.randn())
    smmask[i,:,:,:] = scipy.ndimage.gaussian_filter(masks[i,:,:,:], sm)
    smmask[i,:,:,:] = numpy.where(smmask[i,:,:,:] > numpy.quantile(smmask[i,:,:,:], 0.925), 1, 0).astype(float)

numpy.save(os.path.join('/data', '1000_lesions_3D.npy'), smmask)

# ---- 4. 3D disconnectome ---- #


# mask='/data/two_point_substrate_3D_bin.nii.gz'
# tckedit $TEMPLATEDIR/Tractograms/dTOR_full_tractogram.tck -number 100K /data/dTOR_100K.tck
# tckedit /data/dTOR_100K.tck -include /data/two_point_substrate_3D_bin.nii.gz /data/two_dot_tracts.tck

tracts = nibabel.load(os.path.join('/data', 'two_point_substrate_3D_tracts.nii')).get_fdata()
res_tracts = resize(tracts, (32,32,32))

nibabel.save(nibabel.Nifti1Image(res_tracts, numpy.eye(4)), os.path.join('/data', 'two_point_substrate_3D_tracts_32.nii.gz'))


# ---- 5. create new lesion blobs ---- #

Path = '/mnt/h/DLDM'

dim = (32,32,32)

empty = numpy.zeros(dim)


for l in range(10):
    print(f"Creating lesion {l}")
    try:
        tmp = empty.copy()
        tmp[numpy.random.randint(0,32),numpy.random.randint(0,32),numpy.random.randint(0,32)] = 1
        idx = numpy.where(tmp ==1)
        idx_new = (idx[0]+numpy.random.randint(3),idx[1]+numpy.random.randint(3),idx[2]+numpy.random.randint(3))
        for i in (0,1,2):
            if idx_new[i]>32:
                idx_new[i]=32
        tmp[idx_new[0], idx_new[1], idx_new[2]] = 1
        sm =  numpy.random.rand()
        smtmp = scipy.ndimage.gaussian_filter(tmp, float(sm))
        smtmp = numpy.where(smtmp > numpy.quantile(smtmp, 0.75), 1, 0).astype(float)
        idx = numpy.where(smtmp >0)
        rand_order = numpy.random.permutation(idx[0].shape[0])
        tmp = empty.copy()
        for i in (0,1,2):
            tmp[idx[0][rand_order[i]-1], idx[1][rand_order[i]-1], idx[2][rand_order[i]-1]] = 1
        tmp = scipy.ndimage.gaussian_filter(tmp, 0.2)
        smtmp = smtmp - numpy.where(tmp > numpy.quantile(tmp, 0.95), 1, 0)
        smtmp = scipy.ndimage.gaussian_filter(smtmp, 0.1)
        smtmp = numpy.where(smtmp > numpy.quantile(smtmp, 0.95), 1, 0).astype(float)
        # del_size = numpy.round(numpy.random.rand()*len(rand_order),0).astype(int)
        # for i in range(del_size):
        #     smtmp[idx[0][rand_order[i]], idx[1][rand_order[i]], idx[2][rand_order[i]]] = 0
        # smtmp = scipy.ndimage.gaussian_filter(smtmp, 0.5)
        # smtmp = numpy.where(smtmp > numpy.quantile(smtmp, 0.95), 1, 0).astype(float)
        nii = nibabel.Nifti1Image(smtmp, numpy.eye(4))
        nibabel.save(nii, os.path.join(Path, f'new_lesion_{l}.nii.gz'))
    except Exception as e:
        print(f"Error creating lesion {l}: {e}")
        continue

        


'''


# ---- 5. create new lesion blobs ---- #

# Path = '/mnt/h/DLDM'
# Path = '/Users/patrikbey/Data'
# Path = '/data'

import numpy, nibabel, scipy.ndimage, progress.bar, os, matplotlib.pyplot as plt
from monai.transforms import Compose, Resize

# def random_blob(shape=(32,32,32), min_points=1, max_points=5, max_radius=5):
#     arr = numpy.zeros(shape)
#     # Randomly choose number of seed points
#     n_points = numpy.random.randint(min_points, max_points+1)
#     for _ in range(n_points):
#         # Random center
#         center = [numpy.random.randint(0, s) for s in shape]
#         # Random radius
#         radius = numpy.random.randint(1, max_radius)
#         # Create a spherical blob
#         zz, yy, xx = numpy.ogrid[:shape[0], :shape[1], :shape[2]]
#         mask = (zz-center[0])**2 + (yy-center[1])**2 + (xx-center[2])**2 <= radius**2
#         arr[mask] = 1
#     # Optionally smooth to get more organic shapes
#     arr = scipy.ndimage.gaussian_filter(arr, sigma=numpy.random.uniform(0.5, 2.5))
#     # Threshold to get binary blob
#     arr = (arr > arr.mean()).astype(float)
#     return arr

def get_radius(mu = 4, sigma = 2):
    '''
    return positive random radius for lesion blob
    initialization
    '''
    for i in range(100):
        rn = numpy.random.normal(mu,sigma)
        if rn >0:
            return(rn)


def resize(volume, target_size):
    resize_transform = Compose([Resize((target_size[0],
                                        target_size[1],
                                        target_size[2]))])
    if len(volume.shape) == 3:
        volume = numpy.expand_dims(volume, axis=0)
    resized_volume = resize_transform(volume)
    resized_volume = numpy.squeeze(resized_volume)
    return resized_volume


def random_blob(shape=(32,32,32), min_radius=1, max_radius=6):
    """
    Create a single random spherical blob in a numpy array.
    """
    arr = numpy.zeros(shape)
    # Random center
    center = [numpy.random.randint(0, s) for s in shape]
    # Random radius
    # chance = numpy.random.random()
    # if chance > .66:
    #     max_radius = 3
    # # radius = numpy.random.randint(min_radius, max_radius+1)
    radius = get_radius()
    # Create a spherical blob
    zz, yy, xx = numpy.ogrid[:shape[0], :shape[1], :shape[2]]
    mask = (zz-center[0])**2 + (yy-center[1])**2 + (xx-center[2])**2 <= radius**2
    arr[mask] = 1
    # Optionally smooth to get more organic shapes
    arr = scipy.ndimage.gaussian_filter(arr, sigma=numpy.random.uniform(0.5, 2.5))
    # Threshold to get binary blob
    arr = (arr > arr.mean()).astype(float)
    return arr


def adjust_shape(blob):
    '''
    randomly remove area from blob do create random mask shapes
    '''
    chance = numpy.random.random()
    dim = blob.shape
    if chance > .5:
        tmp = numpy.zeros(dim)
        idx = numpy.where(blob == 1)
        reduce_count = numpy.round(get_radius(2.5,1),0).astype(int)
        for i in range(reduce_count):
            rng_idx = numpy.random.randint(0, idx[0].shape[0])
            idx_new = (idx[0][rng_idx],idx[1][rng_idx],idx[2][rng_idx])
            tmp[idx_new[0], idx_new[1], idx_new[2]] = 1
            sm =  get_radius(1,1)
            smtmp = scipy.ndimage.gaussian_filter(tmp, float(sm))
            smtmp = ( smtmp > smtmp.mean()).astype(float)
            blob=blob - smtmp
        out = (blob > 0).astype(float)
        return(out)
    else:
        return(blob)


# # template_brain = numpy.load(os.path.join(Path,'MNI152_T1_32.npy'))

# full = nibabel.load(os.path.join(Path,'full.nii')).get_fdata()
# mask = nibabel.load(os.path.join(Path,'mask.nii')).get_fdata()
# # idx = numpy.mean(numpy.where(mask > 0),axis = 1).astype(int)

# # la = numpy.zeros(full.shape)
# # la[idx[0], idx[1], idx[2]] = 25
# # smmask = scipy.ndimage.gaussian_filter(la, 50)

# # nibabel.save(nibabel.Nifti1Image(smmask, numpy.eye(4)), os.path.join(Path, 'smmask.nii.gz'))

# # smmask = smmask / smmask.max()

# brain = full*mask
# brain = nibabel.load(os.path.join(Path,'Templates','mni152.nii')).get_fdata()
# # brain = brain*smmask

# numpy.min(brain[brain>0])
# th = numpy.quantile(brain[brain>0], 0.05)
# brain = numpy.where(brain <=th,th, brain)
# brain = brain - th

# brain_rs = resize(brain, (32,32,32))

# nibabel.save(nibabel.Nifti1Image(brain_rs, numpy.eye(4)), os.path.join(Path, 'brain_resized.nii.gz'))

# for i in range(10):
#     blob = random_blob()
#     tmp = template_brain * blob
#     nii = nibabel.Nifti1Image(tmp, numpy.eye(4))
# #     nibabel.save(nii, os.path.join(Path, f'new_lesion_{i}.nii.gz'))

# brain_rs = resize(nibabel.load(os.path.join(Path, 'brain_resized.nii.gz')).get_fdata(),(32,32,32))

# la = numpy.where(brain_rs>numpy.quantile(brain_rs[brain_rs>0],0.1),brain_rs,0)
# nii = nibabel.Nifti1Image(la, numpy.eye(4))
# nibabel.save(nii, os.path.join(Path, f'brain_32_rescale.nii.gz'))

# brain_rs = resize(nibabel.load(os.path.join(Path, 'brain_32_rescale.nii.gz')).get_fdata(),(32,32,32))
brain_rs = numpy.load(os.path.join(Path,'mni_brain_32.npy'))


n_runs = 30000
n_masks = 5000
masks = numpy.zeros([n_masks, 32, 32, 32])
idx = 0


with progress.bar.Bar(f'creating lesion masks', max = n_masks) as bar:
    for i in range(n_runs):
        while idx < n_masks:
            blob = random_blob()
            # updated_blob = adjust_shape(blob)
            # tmp = brain_rs * updated_blob
            tmp = brain_rs * blob
            # chance = numpy.random.random()
            # if chance > .5:
                # mask = numpy.where(tmp > numpy.quantile(brain_rs, 0.8), 1, 0).astype(float)
            if tmp.max() > 0:
                mask = numpy.where(tmp > numpy.quantile(tmp[tmp>0], 0.5), 1, 0).astype(float)
                if mask.sum() > 10:
                    masks[idx,:,:,:] = mask
                    idx+=1
                    if idx % 1000 == 0:
                        nii = nibabel.Nifti1Image(mask, numpy.eye(4))
                        nibabel.save(nii, os.path.join(Path, f'new_lesion_{idx}.nii.gz'))
                    bar.next()


# masks = masks[:idx,:,:,:]

la = numpy.sum(masks, axis = 0)
nii = nibabel.Nifti1Image(la, numpy.eye(4))
nibabel.save(nii, os.path.join(Path, f'{n_masks}_lesion_aggregate.nii.gz'))

# agg2D = numpy.rot90(numpy.sum(la, axis = 0),1)
# plt.imshow(numpy.rot90(brain_rs[16,:,:],1), cmap = 'gray')
# plt.imshow(numpy.where(agg2D>0,agg2D,numpy.nan), alpha = 0.5, cmap = 'jet')
# plt.colorbar()
# plt.show()


# numpy.save(os.path.join(Path, '20K_lesions.npy'), masks)

# smmask = numpy.zeros(masks.shape)

# with progress.bar.Bar(f'updating lesion masks', max = n_masks) as bar:
#     for l in range(n_masks):
#         tmp = masks[l,:,:,:]
#         idx = numpy.where(tmp == 1)
#         for i in range(numpy.random.randint(1, 5)):
#             idx_point = numpy.random.randint(0,len(idx[0]))
#             empty = numpy.zeros(tmp.shape)
#             empty[idx[0][idx_point], idx[1][idx_point], idx[2][idx_point]] = 1
#             sm = scipy.ndimage.gaussian_filter(empty, 0.33*numpy.random.rand())
#             tmp = sm+tmp
#         smmask[l,:,:,:] = numpy.where(tmp > numpy.quantile(tmp, 0.75), 1, 0).astype(float)
#         bar.next()

numpy.save(os.path.join(Path, f'{n_masks}_lesions_3D.npy'), masks)

# la = numpy.sum(smmask, axis = 0)
# nii = nibabel.Nifti1Image(la, numpy.eye(4))
# nibabel.save(nii, os.path.join(Path, f'20K_lesion_aggregate_sm.nii.gz'))


# agg2D = numpy.rot90(numpy.sum(la, axis = 0),1)
# plt.imshow(numpy.where(agg2D>0,agg2D,numpy.nan))
# plt.colorbar()
# plt.show()
###################################
#                                 #
#        CREATE 2D LESIONS.       # 
#                                 #
###################################

# n_masks = 20000
# n_masks = 5000


lesions = numpy.zeros([n_masks, 32, 32])

for l in range(len(masks)):
    tmp = numpy.sum(masks[l,:,:,:], axis = 0)
    tmp = numpy.where(tmp>0,1,0)
    # lesions[l,:, :] = numpy.rot90(tmp[numpy.median(idx).astype(int),:,:],1)
    lesions[l] = numpy.rot90(tmp,1)




examples = numpy.random.permutation(n_masks)[:20]
la = numpy.sum(lesions, axis = 0)

for i in range(len(examples)):
    plt.subplot(5,4,i+1)
    plt.imshow(la, cmap='gray', alpha = .5)
    plt.imshow(numpy.where(lesions[examples[i],:,:]==1,1,numpy.nan), cmap='plasma')



plt.suptitle('2D lesion examples')
plt.savefig(os.path.join(Path, f'{n_masks}_lesion_examples_2D.png'))
plt.close()

volumes = numpy.sum(lesions, axis = (1,2))
plt.hist(volumes, bins = 100)
plt.savefig(os.path.join(Path, f'{n_masks}_lesion_volume_dist.png'))
plt.close()



numpy.save(os.path.join(Path, f'{n_masks}_lesions_2D.npy'), lesions)





la = numpy.sum(lesions, axis = 0)
plt.imshow(numpy.where(la>0,la,numpy.nan))
plt.colorbar()
plt.savefig(os.path.join(Path, f'{n_masks}_lesion_aggregate_2D.png'))
plt.close()
# plt.show()


# masks = numpy.load(os.path.join(Path, '5000_lesions_smmask.npy'))
# lesions = numpy.sum(masks, axis = 1)


# lesions = numpy.load('20000_lesions_2D.npy')

# qm = numpy.load(os.path.join('question_mark_substrate_mapped.npy'))

# tmp = numpy.sum(lesions, axis = 0)

# plt.imshow(numpy.where(tmp>0,1,0), cmap = 'gray')
# plt.imshow(qm, cmap = 'plasma', alpha = .5)
# plt.show()

# uqm = numpy.concatenate([qm,numpy.zeros([32,2])], axis = 1)
# uqm = numpy.delete(uqm,0, axis = 1)

# plt.imshow(numpy.where(la>0,la,numpy.nan), cmap = 'gray')
# plt.imshow(numpy.where(qm>0,1,numpy.nan), alpha = .75)
# plt.show()

# m = numpy.where(la>0,1,0)
# qm = m*uqm



'''

TEMPLATE BRAIN MASK

'''
mni = nibabel.load('/data/mni152.nii.gz').get_fdata()

mni = resize(mni,[32,32,32])
numpy.save('/data/mni_brain_32.npy',mni)

template_brain = mni[16,:,:]

# uqm = numpy.delete(uqm,(32), axis = 0)
# uqm[28,:] = 0
# numpy.save('question_mark_substrate_new.npy',qm)

plt.subplot(1,3,1)
plt.imshow(numpy.where(la>0,la,numpy.nan), cmap = 'gray')
plt.imshow(numpy.where(uqm>0,1,numpy.nan), alpha = .75)
plt.subplot(1,3,2)
plt.imshow(numpy.where(la>0,la,numpy.nan), cmap = 'gray')
plt.imshow(numpy.where(dd>0,1,numpy.nan), alpha = .75)
plt.subplot(1,3,3)
plt.imshow(numpy.where(la>0,la,numpy.nan), cmap = 'gray')
plt.imshow(numpy.where(cc>0,1,numpy.nan), alpha = .75)

plt.subplot(2,3,5)
plt.imshow(numpy.where(la>0,la,numpy.nan), cmap = 'gray')
plt.imshow(qm)
plt.subplot(2,3,3)
plt.imshow(numpy.where(la>0,la,numpy.nan), cmap = 'gray')
plt.imshow(qm)
plt.subplot(2,3,6)
plt.imshow(numpy.where(la>0,la,numpy.nan), cmap = 'gray')
plt.imshow(qm)

# ---- compute deficit scores for 2D lesions ---- #

deficit_scores_qm = []
deficit_scores_dd = []
deficit_scores_ze = []


for l in range(n_masks):
    tmp = lesions[l,:,:]
    overlap_qm = tmp * qm
    overlap_dd = tmp * dd
    overlap_ze = tmp * ze
    deficit_scores_qm.append(numpy.count_nonzero(overlap_qm))
    deficit_scores_dd.append(numpy.count_nonzero(overlap_dd))
    deficit_scores_ze.append(numpy.count_nonzero(overlap_ze))


plt.hist(deficit_scores_qm, bins = 100, label = 'question mark', alpha = .5)
plt.hist(deficit_scores_dd, bins = 100, label = 'Z', alpha = .5)
plt.hist(deficit_scores_ze, bins = 100, label = 'zero', alpha = .5)
plt.title('deficit scores for 2D lesions - question mark')
plt.xlabel('deficit score')
plt.ylabel('lesion count')
plt.legend()
plt.show()

# idx_new = (idx[0]+numpy.random.randint(3),idx[1]+numpy.random.randint(3),idx[2]+numpy.random.randint(3))
# for i in (0,1,2):
#     if idx_new[i]>32:
#         idx_new[i]=32
# tmp[idx_new[0], idx_new[1], idx_new[2]] = 1
# sm =  numpy.random.rand()
# smtmp = scipy.ndimage.gaussian_filter(tmp, float(sm))
# smtmp = numpy.where(smtmp > numpy.quantile(smtmp, 0.75), 1, 0).astype(float)
# idx = numpy.where(smtmp >0)
# rand_order = numpy.random.permutation(idx[0].shape[0])
# tmp = empty.copy()
# for i in (0,1,2):
#     tmp[idx[0][rand_order[i]-1], idx[1][rand_order[i]-1], idx[2][rand_order[i]-1]] = 1
# tmp = scipy.ndimage.gaussian_filter(tmp, 0.2)
# smtmp = smtmp - numpy.where(tmp > numpy.quantile(tmp, 0.95), 1, 0)
# smtmp = scipy.ndimage.gaussian_filter(smtmp, 0.1)
# smtmp = numpy.where(smtmp > numpy.quantile(smtmp, 0.95), 1, 0).astype(float)
# # del_size = numpy.round(numpy.random.rand()*len(rand_order),0).astype(int)
# # for i in range(del_size):
# #     smtmp[idx[0][rand_order[i]], idx[1][rand_order[i]], idx[2][rand_order[i]]] = 0
# # smtmp = scipy.ndimage.gaussian_filter(smtmp, 0.5)
# # smtmp = numpy.where(smtmp > numpy.quantile(smtmp, 0.95), 1, 0).astype(float)
# nii = nibabel.Nifti1Image(smtmp, numpy.eye(4))
# nibabel.save(nii, os.path.join(Path, f'new_lesion_{l}.nii.gz'))
# except Exception as e:
# print(f"Error creating lesion {l}: {e}")
# continue

        

###################################
#                                 #
#        COMPUTE DEFICITS         #
#                                 #
###################################


# --- 1. 3D question mask - binary overlay --- #

# img = nibabel.load(os.path.join('/data', 'question_mark_substrate_3D_sm.nii.gz')).get_fdata()


# # ---- 1.1 fake lesion blobs ---- #
# lesions1000 = numpy.load(os.path.join('/data', '1000_lesions_3D.npy'))
# deficit_scores_bin = get_deficit_score(lesions1000, img) # 384 lesions > 5% == 2D
# deficit_scores_cont = get_deficit_score(lesions1000, img, binary=False)

# # ---- 1.2 real lesion blobs ---- #
# lesions = numpy.load(os.path.join(os.getenv('TEMPLATEDIR'), '1500_lesions.npy'))
# deficit_scores = get_deficit_score(lesions, img) # 81 lesions > 5%


# # ---- 2. 3D two point - binary overlay ---- #

# img = nibabel.load(os.path.join('/data', 'two_point_substrate_3D_bin.nii.gz')).get_fdata()

# # ---- 2.1 fake lesion blobs ---- #
# lesions1000 = numpy.load(os.path.join('/data', '1000_lesions_3D.npy'))
# deficit_scores = get_deficit_score(lesions1000, img) # 400 lesions > 5% == 2D

# # ---- 2.2 real lesion blobs ---- #
# lesions = numpy.load(os.path.join(os.getenv('TEMPLATEDIR'), '1500_lesions.npy'))
# deficit_scores = get_deficit_score(lesions, img) # 80 lesions > 5%



###################################
#                                 #
#      PHYSSTROKE DATA PREP       #
#                                 #
###################################



import os, numpy, nibabel, progress.bar



def resize(volume, target_size):
    import numpy
    from monai.transforms import Compose, Resize
    resize_transform = Compose([Resize((target_size[0],
                                        target_size[1],
                                        target_size[2]))])
    if len(volume.shape) == 3:
        volume = numpy.expand_dims(volume, axis=0)
    resized_volume = resize_transform(volume)
    resized_volume = numpy.squeeze(resized_volume)
    return(resized_volume)

Path = '/mnt/h/SPRING-AI/Data'


subject_masks= [ f if f.startswith('sub-') else None for f in os.listdir(os.path.join(Path,'MNIMasks','PhysStroke')) ]

subjects = [ f.split('_')[0] for f in subject_masks ]

deficit_scores = []

lesions = numpy.zeros([len(subjects), 32,32,32])
subcount = 0

with progress.bar.Bar('Preparing lesion data', max = len(subjects)) as bar:
    for sub in subjects:
        barthelfile = os.path.join(Path, 'ClinicalScoresBIDS_all', sub, 'ses-01', 'phenotype', 'Barthel.tsv')
        if os.path.exists(barthelfile):
            tmp = numpy.genfromtxt(barthelfile, dtype=str)[1,1].astype(int)
            deficit_scores.append((100-tmp)/100)
            # mask = nibabel.load(os.path.join(Path,'MNIMasks','PhysStroke', f'{sub}_MNI_lesion_mask.nii.gz')).get_fdata()
            # tmp = resize(mask, (32,32,32))
            # lesions[subcount] = tmp
            subcount+=1
        else:
            print(f"Warning: {sub} Barthel not found.")
        bar.next()



# ---- 2D ---- #

import os, numpy, progress.bar, matplotlib.pyplot as plt
Path = '/mnt/h/DLDM/Testing/28-08-2025/PhysStroke'
lesions = numpy.load(os.path.join(Path, 'PhysStroke_lesions_resize.npy'))

lesions = numpy.sum(lesions, axis = 1)
for i in range(lesions.shape[0]):
    lesions[i] = numpy.rot90(lesions[i],1)
    lesions[i] = numpy.where(lesions[i]>0,1,0)

agg = numpy.sum(lesions, axis = 0)
agg = numpy.rot90(agg,1)


for i in range(15):
    plt.subplot(3,5,i+1)
    plt.imshow(numpy.where(agg>0,1,0))
    plt.imshow(lesions[i], alpha = .5, cmap = 'grey')


numpy.save(os.path.join(Path, f'PhysStroke_lesions_2D.npy'), lesions)
numpy.save(os.path.join(Path, f'PhysStroke_deficit_scores.npy'), deficit_scores)










###################################
#                                 #
#          2D DATA PREP           #
#                                 #
###################################


import os, numpy, nibabel, progress.bar



def resize(volume, target_size):
    import numpy
    from monai.transforms import Compose, Resize
    resize_transform = Compose([Resize((target_size[0],
                                        target_size[1],
                                        target_size[2]))])
    if len(volume.shape) == 3:
        volume = numpy.expand_dims(volume, axis=0)
    resized_volume = resize_transform(volume)
    resized_volume = numpy.squeeze(resized_volume)
    return(resized_volume)

Path = '/mnt/h/SPRING-AI/Data'


subject_masks= [ f if f.startswith('sub-') else None for f in os.listdir(os.path.join(Path,'MNIMasks','PhysStroke')) ]

subjects = [ f.split('_')[0] for f in subject_masks ]

deficit_scores = []

lesions = numpy.zeros([len(subjects), 32,32,32])
subcount = 0

with progress.bar.Bar('Preparing lesion data', max = len(subjects)) as bar:
    for sub in subjects:
        barthelfile = os.path.join(Path, 'ClinicalScoresBIDS_all', sub, 'ses-01', 'phenotype', 'Barthel.tsv')
        if os.path.exists(barthelfile):
            tmp = numpy.genfromtxt(barthelfile, dtype=str)[1,1].astype(int)
            deficit_scores.append(tmp)
            mask = nibabel.load(os.path.join(Path,'MNIMasks','PhysStroke', f'{sub}_MNI_lesion_mask.nii.gz')).get_fdata()
            tmp = resize(mask, (32,32,32))
            lesions[subcount] = tmp
            subcount+=1
        else:
            print(f"Warning: {sub} Barthel not found.")
        bar.next()

def adjust_shape(blob):
    '''
    randomly remove area from blob do create random mask shapes
    '''
    chance = numpy.random.random()
    dim = blob.shape
    if chance > .33:
        tmp = numpy.zeros(dim)
        idx = numpy.where(blob == 1)
        for i in range(numpy.random.randint(1,4)):
            rng_idx = numpy.random.randint(0, idx[0].shape[0])
            idx_new = (idx[0][rng_idx],idx[1][rng_idx],idx[2][rng_idx])
            tmp[idx_new[0], idx_new[1], idx_new[2]] = 1
            sm =  numpy.random.rand()
            smtmp = scipy.ndimage.gaussian_filter(tmp, float(sm))
            smtmp = ( smtmp > smtmp.mean()).astype(float)
            blob=blob - smtmp
        out = (blob > 0).astype(float)


n_masks = 5000
lesions = numpy.zeros([n_masks, 32, 32])

for l in range(len(masks)):
    tmp = masks[l,:,:,:]
    idx = numpy.where(tmp ==1)[0]
    lesions[l,:, :] = numpy.rot90(tmp[numpy.median(idx).astype(int),:,:],1)


numpy.save(os.path.join(Path, f'{n_masks}_lesions_2D.npy'), lesions)

la = numpy.sum(lesions, axis = 0)
plt.imshow(la)
plt.colorbar()
plt.show()

qm = numpy.load(os.path.join('Docker','data','toy_examples','question_mark_substrate.npy'))
dd = numpy.load(os.path.join('Docker','data','toy_examples','double_dot_substrate.npy'))
ze = numpy.load(os.path.join('Docker','data','toy_examples','zero_substrate.npy'))

# ---- compute deficit scores for 2D lesions ---- #

deficit_scores_qm = []
deficit_scores_dd = []
deficit_scores_ze = []


for l in range(n_masks):
    tmp = lesions[l,:,:]
    overlap_qm = tmp * qm
    overlap_dd = tmp * dd
    overlap_ze = tmp * ze
    deficit_scores_qm.append(numpy.count_nonzero(overlap_qm))
    deficit_scores_dd.append(numpy.count_nonzero(overlap_dd))
    deficit_scores_ze.append(numpy.count_nonzero(overlap_ze))


plt.hist(deficit_scores_qm, bins = 100, label = 'question mark', alpha = .5)
plt.hist(deficit_scores_dd, bins = 100, label = 'Z', alpha = .5)
plt.hist(deficit_scores_ze, bins = 100, label = 'zero', alpha = .5)
plt.title('deficit scores for 2D lesions - question mark')
plt.xlabel('deficit score')
plt.ylabel('lesion count')
plt.legend()
plt.show()




#######################
#                     #
# AAL3 motor substrate#
#                     # 
#######################


aal3 = nibabel.load('AAL3_MNI152.nii.gz').get_fdata()

# AAL3 motor areas relevant in Stroke (Assis et al. (2023)):

# Motor areas 
# 1, 2 Precentral_L, R 
# 15, 16 Supp_Motor_Area_L, R 
# Cerebellum 
# 95, 96 Cerebellum_Crus1_L, R 
# 97, 98 Cerebellum_Crus2_L, R 
# 99, 100 Cerebellum_3_L, R 
# 101, 102 Cerebellum_4_5_L, R 
# 103, 104 Cerebellum_6_L, R 
# 105, 106 Cerebellum_7b_L, R 
# 107, 108 Cerebellum_8_L, R 
# 109, 110 Cerebellum_9_L, R 
# 111, 112 Cerebellum_10_L, R 

motor = numpy.zeros(aal3.shape)

for i in [1,2,15,16,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112]:
    motor = numpy.where(aal3 == i,1,0) + motor

nii = nibabel.Nifti1Image(motor.astype(float), affine=nibabel.load('AAL3_MNI152.nii.gz').affine)

nibabel.save(nii, 'AAL3_motor_substrate.nii.gz')

nii_rs = resize(motor, (32,32,32))

nii = nibabel.Nifti1Image(nii_rs.astype(float), affine=nibabel.load('AAL3_MNI152.nii.gz').affine)
nibabel.save(nii, 'AAL3_motor_substrate_resize.nii.gz')


la = numpy.sum(nii_rs, axis = 0)
la = numpy.rot90(numpy.where(la>0,1,0),1)



# ---- validate lesion overlay using BI ---- 

import os, numpy, nibabel, matplotlib.pyplot as plt

Path = '/mnt/h/DLDM/TESTING/28-08-2025/PhysStroke'

template = numpy.load(os.path.join(Path, 'MNI152_T1_32.npy'))
brain_slice = numpy.rot90(template[16,:,:],1)
BI = numpy.load(os.path.join(Path, 'PhysStroke_deficit_scores.npy'))
lesions = numpy.load(os.path.join(Path, 'PhysStroke_lesions_resize.npy'))
lesions = numpy.sum(lesions, axis = 1)
for l in range(lesions.shape[0]):
    lesions[l] = numpy.rot90(numpy.where(lesions[l]>0,1,0),1)


low = numpy.where(BI <= numpy.quantile(BI, 0.33))[0]
mid = numpy.where((BI > numpy.quantile(BI, 0.33)) & (BI <= numpy.quantile(BI, 0.66)))[0]
high = numpy.where(BI > numpy.quantile(BI, 0.66))[0]


plt.subplot(1,3,1)
tmp = numpy.sum(lesions[low,:,:], axis = 0)
plt.imshow(brain_slice, cmap='gray', alpha = .75)
plt.imshow(tmp, cmap='plasma', alpha = .5)
plt.title('low-severity')
plt.subplot(1,3,2)
tmp = numpy.sum(lesions[mid,:,:], axis = 0)
plt.imshow(brain_slice, cmap='gray', alpha = .75)
plt.imshow(tmp, cmap='plasma', alpha = .5)
plt.title('mid-severity')
plt.subplot(1,3,3)
tmp = numpy.sum(lesions[high,:,:], axis = 0)
plt.imshow(brain_slice, cmap='gray', alpha = .75)
plt.imshow(tmp, cmap='plasma', alpha = .5)
plt.title('high-severity')
plt.savefig(os.path.join(Path, 'Lesion_aggregates_BI_sorted.png'))
plt.close()




#########################################
#                                       #
#             NEUROQUERY                #
#                                       #
#########################################

import os, numpy, nibabel, matplotlib.pyplot as plt

Path = '/mnt/h/DLDM/substrates'

motor = 'arm_motor.nii.gz'
cog = 'coordination_speech_inattention.nii.gz'


motor_img = nibabel.load(os.path.join(Path, motor)).get_fdata()
cog_img = nibabel.load(os.path.join(Path, cog)).get_fdata()

motor_img = resize(motor_img, (32,32,32))
cog_img = resize(cog_img, (32,32,32))

cog_sum = numpy.where(numpy.rot90(numpy.sum(cog_img, axis = 0),1)>10,1,0)
motor_sum = numpy.where(numpy.rot90(numpy.sum(motor_img, axis = 0),1)>10,1,0)

plt.subplot(1,2,1)
plt.imshow(cog_sum, cmap = 'gray')
plt.title('cognition substrate')
plt.subplot(1,2,2)
plt.imshow(motor_sum, cmap = 'gray')
plt.title('motor substrate')
plt.show()

numpy.save(os.path.join(Path, 'cognition_substrate_2D.npy'), cog_sum)
numpy.save(os.path.join(Path, 'motor_substrate_2D.npy'), motor_sum)
#########################################
#                                       #
#                   UCLH                #
#                                       #
#########################################


# import numpy, os

# Path = '/mnt/h/UCLH'


# full = numpy.genfromtxt(os.path.join(Path, 'ssnap_uclh_anon.csv'), delimiter=',', dtype = str)
# full = numpy.genfromtxt(os.path.join(Path, 'ssnap_uclh_anon_MRI.csv'), delimiter=',', dtype = str)

# column_names = list(full[0,:])
# combi_full = [ full[i,1] + full[i,2] for i in range(1,full.shape[0]) ]

# numpy.unique(combi_full, return_counts = True)


# numpy.where(full[0,:] == 'S2BrainImagingNotPerformed')


# numpy.unique(full[1:,column_names.index('S2BrainImagingModality')], return_counts = True)


# numpy.where(full[1:,column_names.index('S2BrainImagingModality')] == 'MRI')



#########################################
#                                       #
#        IN-CONTEXT LEARNING            #
#                                       #
#########################################

# create subset of UCLH patients with significant motor / cognitive deficit
# according to NIHSS catgories
#
# MOTOR DEFICIT NIHSS CATEGORIES:
# S2NihssArrivalMotorArmLeft >= 3
# S2NihssArrivalMotorArmRight >= 3
# S2NihssArrivalMotorLegLeft >= 3
# S2NihssArrivalMotorLegRight >= 3
# NO MOTOR DEFICIT:
# S2NihssArrivalMotorArmLeft = 0 (...)

# Source=/mnt/h/UCLH/UCLH
# Target=/mnt/h/DLDM/ICL/data/UCLH/no-motor_deficit
# Subjects=$( cat no-motor_patients_UCLH.csv )

# for s in $Subjects; do
#     cp ${Source}/"${s}".nii.gz "${Target}/${s}.nii.gz"
# done

import os, numpy, nibabel, matplotlib.pyplot as plt, progress.bar

from monai.transforms import Compose, Resize


def resize(volume, target_size):
    resize_transform = Compose([Resize((target_size[0],
                                        target_size[1],
                                        target_size[2]))])
    if len(volume.shape) == 3:
        volume = numpy.expand_dims(volume, axis=0)
    resized_volume = resize_transform(volume)
    resized_volume = numpy.squeeze(resized_volume)
    return resized_volume

Path = '/mnt/h/DLDM/ICL/data/UCLH/no-motor_deficit'

SubjectList = [ f if f.endswith('.nii.gz') else None for f in os.listdir(Path) ]

lesions = numpy.zeros([len(SubjectList), 32,32,32])

with progress.bar.Bar(f'creating lesion masks', max = len(SubjectList)) as bar:
    for sub in SubjectList:
        subid = sub.split('.nii.gz')[0]
        img = nibabel.load(os.path.join(Path, sub)).get_fdata()
        img_rs = resize(img, (32,32,32))
        lesions[SubjectList.index(sub),:,:,:] = img_rs
        bar.next()

numpy.save('/mnt/h/DLDM/ICL/data/UCLH/no-motor_deficit_lesions.npy', lesions)


# ---- 2D ---- #

lesions = numpy.load('/mnt/h/DLDM/ICL/data/UCLH/motor_deficit_lesions.npy')

lesions = numpy.sum(lesions, axis = 1)

for l in numpy.arange(lesions.shape[0]):
    lesions[l,:,:] = numpy.rot90(numpy.where(lesions[l,:,:]>0,1,0),1)

lesions = numpy.rot90(numpy.sum(lesions, axis = 1),1)

numpy.save('/mnt/h/DLDM/ICL/data/UCLH/motor_deficit_lesions_2D.npy', lesions)

lesions = numpy.load('/mnt/h/DLDM/ICL/data/UCLH/no-motor_deficit_lesions.npy')

lesions = numpy.sum(lesions, axis = 1)

for l in numpy.arange(lesions.shape[0]):
    lesions[l,:,:] = numpy.rot90(numpy.where(lesions[l,:,:]>0,1,0),1)



numpy.save('/mnt/h/DLDM/ICL/data/UCLH/no-motor_deficit_lesions_2D.npy', lesions)



# ---- size-ratio deficit update ---- #
lesions = numpy.load('/mnt/h/DLDM/ICL/data/UCLH/motor_deficit_lesions_2D.npy')
substrate = numpy.load('/mnt/h/DLDM/substrates/motor_substrate_2D.npy')

gt_size = numpy.count_nonzero(substrate)
deficit_scores = []
for l in range(lesions.shape[0]):
    lesions_size = numpy.count_nonzero(lesions[l,:,:])
    deficit_scores.append(lesions_size/gt_size)

# ---- prepare motor tracktography substrate ---- #

import os, numpy, nibabel, matplotlib.pyplot as plt, progress.bar
from monai.transforms import Compose, Resize

def resize(volume, target_size):
    resize_transform = Compose([Resize((target_size[0],
                                        target_size[1],
                                        target_size[2]))])
    if len(volume.shape) == 3:
        volume = numpy.expand_dims(volume, axis=0)
    resized_volume = resize_transform(volume)
    resized_volume = numpy.squeeze(resized_volume)
    return resized_volume

Path = '/mnt/h/DLDM/ICL/data/'

img = nibabel.load(os.path.join(Path, 'CorticoSpinalTract_bin.nii')).get_fdata()

img_rs = resize(img, (32,32,32))
numpy.save(os.path.join(Path, 'CorticoSpinalTract_bin_32.npy'), img_rs)

la = numpy.rot90(numpy.sum(img_rs, axis = 0),1)
plt.imshow(numpy.where(la>0,1,0))
plt.show()

numpy.save(os.path.join(Path, 'CST_2D.npy'), numpy.where(la>0,1,0))