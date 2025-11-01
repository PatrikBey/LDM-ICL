

#########################
#                       #
#      UTILITIES  #
#                       #
#########################

def get_variable(_string):
    '''
    retrieve value of environment variables
    defined during container call
    '''
    import os
    return(os.getenv(_string))



def get_device():
    '''
    return cuda device if GPU available
    '''
    import torch
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_msg(_string):
    '''
    logging function printing date, scriptname & input string to stdout
    '''
    import datetime, os, sys
    print(f'{datetime.date.today().strftime("%a %B %d %H:%M:%S %Z %Y")} {str(os.path.basename(sys.argv[0]))}: {str(_string)}')


def resize(volume, target_size=(64,64,64)):
    '''
    resize a given image to the provided target_size
    '''
    from monai.transforms import Compose, Resize
    import numpy
    resize_transform = Compose([Resize((target_size[0],
                                        target_size[1],
                                        target_size[2]))])
    if len(volume.shape) == 3:
        volume = numpy.expand_dims(volume, axis=0)
    resized_volume = resize_transform(volume)
    resized_volume = numpy.squeeze(resized_volume)
    return resized_volume

#########################
#                       #
#       DATASET          #
#                       #
#########################

from torch.utils.data import Dataset  

# class DeficitDataset(Dataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, index):
#         import numpy
#         img = self.data[index]
#         return img, numpy.expand_dims(self.labels[index], axis=0)


class LesionDataset(Dataset):
    '''
    Dataset class containing:
        data = lesion masks
    '''
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        img = self.data[index]
        return img
    @property
    def n_samples(self):
        return(self.data.shape[0])
        



#########################
#                       #
#    VISUALIZATIONS     #
#                       #
#########################

# def visualize_inference3D(gt, rec, template, filename='/data/gt_overlay.png', slice_axis=2, slice_idx=None):
#     '''
#     visualize overlay of ground truth neural substrate with inferred reconstruction for 3D data
#     Shows a 2D slice of the 3D volume for visualization
#     :param gt: ground truth 3D array
#     :param rec: reconstructed 3D array  
#     :param template: template 3D brain array
#     :param filename: output filename
#     :param slice_axis: axis along which to slice (0=sagittal, 1=coronal, 2=axial)
#     :param slice_idx: specific slice index, if None uses middle slice
#     '''
#     import numpy, matplotlib.pyplot as plt
    
#     # Select middle slice if not specified
#     if slice_idx is None:
#         slice_idx = template.shape[slice_axis] // 2
    
#     # Extract 2D slices based on axis
#     if slice_axis == 0:  # sagittal
#         template_slice = template[slice_idx, :, :]
#         gt_slice = gt[slice_idx, :, :] if len(gt.shape) == 3 else gt
#         rec_slice = rec[slice_idx, :, :] if len(rec.shape) == 3 else rec
#     elif slice_axis == 1:  # coronal
#         template_slice = template[:, slice_idx, :]
#         gt_slice = gt[:, slice_idx, :] if len(gt.shape) == 3 else gt
#         rec_slice = rec[:, slice_idx, :] if len(rec.shape) == 3 else rec
#     else:  # axial (default)
#         template_slice = template[:, :, slice_idx]
#         gt_slice = gt[:, :, slice_idx] if len(gt.shape) == 3 else gt
#         rec_slice = rec[:, :, slice_idx] if len(rec.shape) == 3 else rec
    
#     plt.imshow(template_slice, cmap='gray')
#     plt.imshow(numpy.where(gt_slice > 0, 1, numpy.nan), cmap='hot')
#     plt.imshow(rec_slice, cmap='plasma', alpha=0.5)
#     plt.colorbar()
#     plt.title(f'3D Inference Visualization - Slice {slice_idx} (axis {slice_axis})')
#     plt.savefig(filename)
#     plt.close()

def visualize_inference3D(gt, rec, template, filename='/data/gt_overlay.png', ):
    '''
    Visualize 3D mask overlay to template brain across axes
    '''
    import numpy, matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    idx = numpy.where(gt > 0)
    idx = numpy.mean(numpy.array(idx), axis=1).astype(numpy.int32)
    plt.subplot(1,3,1)
    plt.imshow(numpy.rot90(numpy.where(template[idx[0],:,:] > 0, template[idx[0],:,:], numpy.nan),1), cmap='gray')
    plt.imshow(numpy.rot90(numpy.where(gt[idx[0],:,:] > 0, 1, numpy.nan),1), cmap='binary', alpha = .5)
    plt.imshow(numpy.rot90(rec[idx[0],:,:],1), cmap='plasma', alpha=0.5)
    plt.title('Sagittal View')
    plt.subplot(1,3,2)
    plt.imshow(numpy.rot90(numpy.where(template[:,idx[1],:] > 0, template[:,idx[1],:], numpy.nan),1), cmap='gray')
    plt.imshow(numpy.rot90(numpy.where(gt[:,idx[1],:] > 0, 1, numpy.nan),1), cmap='binary', alpha = .5)
    plt.imshow(numpy.rot90(rec[:,idx[1],:],1), cmap='plasma', alpha=0.5)
    plt.title('Coronal View')
    plt.subplot(1,3,3)
    plt.imshow(numpy.rot90(numpy.where(template[:,:,idx[0]] > 0, template[:,:,idx[0]], numpy.nan),1), cmap='gray')
    plt.imshow(numpy.rot90(numpy.where(gt[:,:,idx[0]] > 0, 1, numpy.nan),1), cmap='binary', alpha = .5)
    plt.imshow(numpy.rot90(rec[:,:,idx[0]],1), cmap='plasma', alpha=0.5)
    plt.title('Axial View')
    plt.savefig(filename)
    plt.close()


# Keep the original 2D function for backward compatibility
def visualize_inference2D(gt, rec, template, filename = '/data/gt_overlay.png'):
    '''
    visualize overlay of ground truth neural substrate with infered reconstruction
    '''
    import numpy, matplotlib.pyplot as plt
    # plt.imshow(numpy.rot90(template[16,:,:],1), cmap = 'gray')
    plt.imshow(template, cmap = 'gray')
    plt.imshow(numpy.where(gt>0,1,numpy.nan), cmap = 'hot')
    plt.imshow(rec, cmap = 'plasma', alpha = 0.5)
    plt.colorbar()
    # ---- save figure ---- #
    plt.savefig(filename)
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#########################
#                       #
#      VALIDATION       #
#                       #
#########################


def vec_dice(array1,array2):
    '''
    compute dice scores between two 4D arrays
    input:
        4D arrays of lesion masks
    output:
        1D vector of dice scores of all masks
    '''
    import numpy
    im1 = numpy.asarray(array1).astype(bool)
    im2 = numpy.asarray(array2).astype(bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    intersection = numpy.logical_and(im1, im2)
    return(2. * intersection.sum(axis=tuple(range(1,4))) / (im1.sum(axis=tuple(range(1,4))) + im2.sum(axis=tuple(range(1,4)))))

# def dice_3D(array1, array2):
#     '''
#     compute dice score for two input 3D arrays
#     '''
#     import numpy
#     if array1.shape != array2.shape:
#         raise ValueError("Shape mismatch: array1 and array2 must have the same shape.")
#     if (numpy.unique(array1).any() not in [0,1] or numpy.unique(array2).any() not in [0,1]):
#         raise ValueError("Input arrays must be binary (contain only 0s and 1s).")
#     intersection = array1 * array2
#     return(2. * intersection.sum() / (array1.sum() + array2.sum()))

def dice_2D(array1, array2):
    '''
    compute dice score for 2D arrays
    '''
    import numpy
    if array1.shape != array2.shape:
        raise ValueError("Shape mismatch: array1 and array2 must have the same shape.")
    if (numpy.unique(array1).any() not in [0,1] or numpy.unique(array2).any() not in [0,1]):
        raise ValueError("Input arrays must be binary (contain only 0s and 1s).")
    intersection = array1 * array2
    return(2. * intersection.sum() / (array1.sum() + array2.sum()))




def get_deficit(_lesions, _substrate = None, _type = 'overlap_binary',_noise=None):
    '''
    return deficit scores for various computation types
    '''
    import numpy, scipy.ndimage, progress.bar
    deficits = []
    if _type == 'overlap_binary':
        with progress.bar.Bar('Processing', max=len(_lesions)) as bar:
            for i in range(len(_lesions)):
                overlap = _lesions[i] * _substrate
                counts = numpy.count_nonzero(overlap)
                voxels_gt = numpy.sum(_substrate)
                ratio_lesion = counts / voxels_gt
                # using minimal overlap ratio of 5%
                if ratio_lesion > 0.05:
                    deficits.append(1)
                else:
                    deficits.append(0)
                bar.next()
        return numpy.array(deficits)
    elif _type == 'overlap_ratio':
        with progress.bar.Bar('Processing', max=len(_lesions)) as bar:
            for i in range(len(_lesions)):
                overlap = _lesions[i] * _substrate
                counts = numpy.count_nonzero(overlap)
                if counts > 0:
                    voxels_mask = numpy.sum(_lesions[i])
                    ratio_lesion = counts / voxels_mask
                    deficits.append(ratio_lesion)
                else:
                    deficits.append(0)
                bar.next()
        return numpy.array(deficits)
    elif _type == 'overlap_ratio_noisy':
        with progress.bar.Bar('Processing', max=len(_lesions)) as bar:
            for i in range(len(_lesions)):
                overlap = _lesions[i] * _substrate
                counts = numpy.count_nonzero(overlap)
                if counts > 0:
                    voxels_mask = numpy.sum(_lesions[i])
                    ratio_lesion = counts / voxels_mask
                    noise = numpy.random.normal(0, _noise)
                    out = ratio_lesion+noise
                    if out > 0:
                        deficits.append(out)
                    else:
                        deficits.append(0)
                else:
                    deficits.append(0)
                bar.next()
        return numpy.array(deficits)
    elif _type == 'distance':
        lbl = scipy.ndimage.label(_substrate)[0]
        groups = numpy.unique(lbl)
        gtcog = numpy.array(scipy.ndimage.center_of_mass(_substrate, lbl, [groups[1:]]))
        with progress.bar.Bar('Processing', max=len(_lesions)) as bar:
            for l in range(len(_lesions)):
                dist = []
                tmp = _lesions[l,:,:]
                for i in groups[1:]:
                    dist.append(numpy.linalg.norm(gtcog[i-1] - numpy.array(scipy.ndimage.center_of_mass(tmp))))
                bar.next()
                deficits.append(numpy.min(dist))
        deficits = numpy.array(deficits)
        deficits = deficits / numpy.max(deficits)
        return numpy.array(deficits)
    elif _type == 'size':
        gt_size = numpy.sum(_substrate)
        with progress.bar.Bar('Processing', max=len(_lesions)) as bar:
            for i in range(len(_lesions)):
                voxels_mask = numpy.sum(_lesions[i])
                deficits.append(voxels_mask/gt_size)
                bar.next()
        deficits = numpy.array(deficits)
        deficits = deficits / numpy.max(deficits)
        return numpy.array(deficits)
    elif _type == 'trans':
        from dipy.align.transforms import AffineTransform3D
        from dipy.align.imaffine import AffineRegistration
        affreg = AffineRegistration(verbosity=0)
        transform = AffineTransform3D()
        with progress.bar.Bar('Processing', max=len(_lesions)) as bar:
            for i in range(len(_lesions)):
                lesion = _lesions[i]
                affine = affreg.optimize(lesion,_substrate, transform, params0=None)
                deficits.append(numpy.linalg.det(affine.affine))
                bar.next()
        deficits = numpy.array(deficits)
        deficits = deficits / numpy.max(deficits)
        return numpy.array(deficits)


# def get_deficit(_lesions, _substrate = None, _type = 'overlap_binary',_noise=None):
#     '''
#     return deficit scores for various computation types
#     '''
#     import numpy, scipy.ndimage
#     deficits = []
#     if _type == 'overlap_binary':
#         for i in range(len(_lesions)):
#             overlap = _lesions[i] * _substrate
#             counts = numpy.count_nonzero(overlap)
#             voxels_gt = numpy.sum(_substrate)
#             ratio_lesion = counts / voxels_gt
#             # using minimal overlap ratio of 5%
#             if ratio_lesion > 0.05:
#                 deficits.append(1)
#             else:
#                 deficits.append(0)
#         return numpy.array(deficits)
#     elif _type == 'overlap_ratio':
#         for i in range(len(_lesions)):
#             overlap = _lesions[i] * _substrate
#             counts = numpy.count_nonzero(overlap)
#             if counts > 0:
#                 voxels_mask = numpy.sum(_lesions[i])
#                 ratio_lesion = counts / voxels_mask
#                 deficits.append(ratio_lesion)
#             else:
#                 deficits.append(0)
#         return numpy.array(deficits)
#     elif _type == 'overlap_ratio_noisy':
#         for i in range(len(_lesions)):
#             overlap = _lesions[i] * _substrate
#             counts = numpy.count_nonzero(overlap)
#             if counts > 0:
#                 voxels_mask = numpy.sum(_lesions[i])
#                 ratio_lesion = counts / voxels_mask
#                 noise = numpy.random.normal(0, _noise)
#                 out = ratio_lesion+noise
#                 if out > 0:
#                     deficits.append(out)
#                 else:
#                     deficits.append(0)
#             else:
#                 deficits.append(0)
#         return numpy.array(deficits)
#     elif _type == 'distance':
#         lbl = scipy.ndimage.label(_substrate)[0]
#         groups = numpy.unique(lbl)
#         gtcog = numpy.array(scipy.ndimage.center_of_mass(_substrate, lbl, [groups[1:]]))
#         for l in range(len(_lesions)):
#             dist = []
#             tmp = _lesions[l,:,:]
#             for i in groups[1:]:
#                 dist.append(numpy.linalg.norm(gtcog[i-1] - numpy.array(scipy.ndimage.center_of_mass(tmp))))
#             deficits.append(numpy.min(dist))
#         deficits = numpy.array(deficits)
#         deficits = deficits / numpy.max(deficits)
#         return numpy.array(deficits)
#     elif _type == 'size':
#         gt_size = numpy.sum(_substrate)
#         for i in range(len(_lesions)):
#             voxels_mask = numpy.sum(_lesions[i])
#             deficits.append(voxels_mask/gt_size)
#         deficits = numpy.array(deficits)
#         deficits = deficits / numpy.max(deficits)
#         return numpy.array(deficits)


