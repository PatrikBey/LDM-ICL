

# ---- create substrate data ---- #
import numpy, nibabel, scipy.ndimage, progress.bar, os, matplotlib.pyplot as plt
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



# ---- load templates ---- #


Path="/mnt/h/ATLAS/MNI"

atlases = ['Schaefer2018','AAL3']


for a in atlases:
    tmp = nibabel.load(os.path.join(Path,f'{a}MNI152.nii.gz')).get_fdata().astype(int)
    rois = numpy.unique(tmp[tmp>0])
    la = numpy.zeros((64, 64, 64))
    with progress.bar.Bar(f'creating rois for {a}', max = len(rois)) as bar:
        for r in rois:
            roi = numpy.where(tmp == r, 1, 0)
            roi_rs = resize(roi, (64,64,64))
            roi_bin = numpy.where(roi_rs > 0.5, 1, 0)
            # roi_bin = numpy.where(roi_rs > numpy.quantile(roi_rs[roi_rs>0], 0.9), 1, 0)
            roi_bin[32:,:,:] = 0
            numpy.save(os.path.join(Path,'substrates',f'{a}_roi_{int(r)}.npy'), roi_bin)
            la = la + roi_bin
            bar.next()
            nibabel.save(nibabel.Nifti1Image(roi_bin.astype(numpy.float32   ), numpy.eye(4)), os.path.join(Path,'nii',f'{a}-{r}.nii.gz'))

# nibabel.save(nibabel.Nifti1Image(la.astype(numpy.float32   ), numpy.eye(4)), os.path.join(Path,f'{a}Combined.nii.gz')) 

for r in numpy.arange(2,171,2):
    file = os.path.join(Path,'substrates',f'{a}_roi_{r}.npy')
    if os.path.exists(file):
        os.remove(file)

# brain = nibabel.load(os.path.join(Path,'MNI152.nii.gz')).get_fdata()

# tmp = resize(brain, (64,64,64))
# nibabel.save(nibabel.Nifti1Image(tmp.astype(numpy.float32   ), numpy.eye(4)), os.path.join(Path,'MNI152_64.nii.gz'))

