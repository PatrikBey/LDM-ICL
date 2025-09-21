#
#
#
# INVESTIGATE DEFICIT SCORE IMPACT ON INFERENCE
#
#

import os, numpy, matplotlib.pyplot as plt

out_dir = '/mnt/h/DLDM/TESTING/31-08-2025/PhysStroke'



substrate = numpy.load(os.path.join(out_dir,'AAL3_motor_substrate_2D.npy'))



lesions = numpy.load(os.path.join(out_dir,'PhysStroke_lesions_resize.npy'))
lesions = numpy.sum(lesions, axis = 1)
for l in range(lesions.shape[0]):
    lesions[l] = numpy.rot90(numpy.where(lesions[l]>0,1,0),1)






barthel_scores = numpy.load(os.path.join(out_dir,'PhysStroke_deficit_scores.npy'))

norm_barthel = barthel_scores / numpy.max(barthel_scores)
from sklearn import preprocessing as pre
barthel_norm = pre.MinMaxScaler().fit_transform(barthel_scores.reshape(-1,1)).reshape(-1)

barthel_binary = numpy.where(barthel_scores > 0.66, 1, 0)

barthel_classes = numpy.digitize(barthel_scores, bins=[0, 0.33, 0.66, 1], right=True)

deficit_ratios = []
for i in range(len(lesions)):
    overlap = lesions[i] * substrate
    counts = numpy.count_nonzero(overlap)
    if counts > 0:
        voxels_mask = numpy.sum(lesions[i])
        ratio_lesion = counts / voxels_mask
        deficit_ratios.append(ratio_lesion)
    else:
        deficit_ratios.append(0)


deficit_scores = numpy.array(deficit_ratios)


plt.subplot(1,2,1)
plt.hist(deficit_scores)
plt.title('Deficit Scores')
plt.subplot(1,2,2)
plt.hist(barthel_scores)
plt.title('Barthel Scores')

plt.show()

plt.plot(deficit_scores, label='Deficit Scores', marker='o')
plt.plot(barthel_scores, label='Barthel Scores', marker='o')
plt.plot(barthel_norm, label='Normalized Barthel Scores', marker='o')
plt.legend()
plt.show()


numpy.corrcoef(deficit_scores, barthel_scores)
numpy.corrcoef(deficit_scores, barthel_norm)
numpy.corrcoef(deficit_scores, barthel_binary)
numpy.corrcoef(deficit_scores, barthel_classes)


