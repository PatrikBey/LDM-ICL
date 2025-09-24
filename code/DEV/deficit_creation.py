#
#
# deficit_creation.py
#
# authors :
# 2. Bey, Patrik
#
#
#
#
# This script contains snippets to
# create the deficit scores for the existing dataset created by data_creation.py:
#
# 1. recon_20K | pretraining dataset used for reconstruction pretraining
# 2. icl_20K | main training dataset used for ICL pretraining
# 3. validation_10K | validation dataset for few-shot training / validation
#
# Deficit scores to be created are:
# 1. Lesion Overlap Ratio (LOR) - continuous score
# 2. Lesion Overlap Binary (LOB) - binary score
# 3. Lesion Distance Score (LDS) - continuous score
# 4. Lesion Size Score (LSS) - continuous score
# 5. Lesion Overlap Ratio with Noise (LORN) - continuous score with noise
# 
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

###################################
#                                 #
#        CREATE DEFICITS          #
#                                 #
###################################


# ---- prepare variables ---- #

Path = '/data/pretrain'
TemplateDir = os.environ.get('TEMPLATEDIR')+'/validation'

datasets = ['recon_20K', 'icl_20K', 'validation_10K']

deficits_types = ['overlap_binary', 'overlap_ratio', 'distance', 'size', 'overlap_ratio_noisy']

substrates = ['CST_2D.npy', 'question_mark_substrate_2D.npy', 'cognition_substrate_2D.npy', 'two_point_substrate_2D.npy', 'motor_substrate_2D.npy', 'circular_substrate_2D.npy', 'question_mark_substrate_2D.npy']

os.makedirs(os.path.join(Path, 'deficits'), exist_ok=True)

for deficit in deficits_types:
    i=0
    plt.figure(figsize=(20,15))
    for substrate in substrates:
        for set in datasets:
            i+=1
            lesions = numpy.load(os.path.join(Path,f'{set}_2D.npy'))
            gt = numpy.load(os.path.join(TemplateDir,substrate))
            score = get_deficit(lesions, _substrate = gt, _type = deficit, _noise=0.05)
            p = plt.subplot(len(datasets),len(substrates),i)
            p = plt.hist(score, bins=20)
            p = plt.title(f'{set} | {substrate[:-4]}')
            p = plt.xlabel('Deficit Score')
            p = plt.ylabel('Counts')
            numpy.save(os.path.join(Path,'deficits',f'{set}_{substrate[:-4]}_{deficit}.npy'), score)
            print(f'Saved: {set}_{substrate[:-4]}_{deficit}.npy')
    plt.tight_layout()
    plt.savefig(os.path.join(Path,'plots',f'{deficit}.png'))
    plt.close()


