

# ---- create training data ---- #
import numpy, nibabel, scipy.ndimage, progress.bar, os, matplotlib.pyplot as plt
from monai.transforms import Compose, Resize


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






# Path="/home/patrik/Data/LDM-ICL"
# Path='/mnt/h/DLDM/ICL/data'
Path='/data'



brain_rs = numpy.load(os.path.join('/data/templates','MNI152_64.npy'))

# ---- create 3D lesion masks ---- #

n_runs = 25000
n_masks = 20000
dims = brain_rs.shape
masks = numpy.zeros([n_masks, *dims])
idx = 0


with progress.bar.Bar(f'creating lesion masks', max = n_masks) as bar:
    for i in range(n_runs):
        while idx < n_masks:
            chance = numpy.random.random()
            if chance > .75:
                blob = random_blob(dims, max_radius=5)
            else:
                blob = random_blob(dims, max_radius=2)
            # updated_blob = adjust_shape(blob)
            # tmp = brain_rs * updated_blob
            tmp = brain_rs * blob
            # chance = numpy.random.random()
            # if chance > .5:
                # mask = numpy.where(tmp > numpy.quantile(brain_rs, 0.8), 1, 0).astype(float)
            if tmp.max() > 0:
                mask = numpy.where(tmp > numpy.quantile(brain_rs[brain_rs>0], 0.8), 1, 0).astype(float)
                if mask.sum() > 10:
                    masks[idx,:,:,:] = mask
                    idx+=1
                    if idx % 1000 == 0:
                        nii = nibabel.Nifti1Image(mask, numpy.eye(4))
                        nibabel.save(nii, os.path.join(Path, f'new_lesion_{idx}.nii.gz'))
                    bar.next()


# ---- save reconstruction pretraining masks ---- #
numpy.save(os.path.join(Path,'pretrain-recon_10K.npy'), masks[:10000,:,:,:])

# ---- save in-context learning pretraining masks ---- #
numpy.save(os.path.join(Path,'pretrain-tune_5K.npy'), masks[10000:15000,:,:,:])

# ---- save fine-tuning masks ---- #
numpy.save(os.path.join(Path,'predict_5K.npy'), masks[15000:,:,:,:])



# ---- CREATE 2D MASKS ---- #

Path='/data/lesions'

masks = os.listdir(Path)


for m in masks:
    tmp = numpy.load(os.path.join(Path,m))
    lesions = numpy.zeros([tmp.shape[0],64,64])
    for i in range(tmp.shape[0]):
        lesions[i] = numpy.where(numpy.rot90(numpy.sum(tmp[i], axis=0), k=1) > 0, 1, 0)
    numpy.save(os.path.join(Path, m.replace('.npy','_2D.npy')), lesions)
    print(f'finished {m}')