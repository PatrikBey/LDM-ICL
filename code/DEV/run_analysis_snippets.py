#
# run analysis template
#
# using 1Kx1K example runs


# Path='/home/pbey/data/outputs/'
# docker run -it -v /home/pbey/data/outputs/:/data/ patrikneuro/dldm:dev python3


import os, numpy, matplotlib.pyplot as plt

Path='/mnt/h/DLDM/VALIDATION'

runs = [f for f in os.listdir(Path) if f.startswith('out-') and os.path.isdir(os.path.join(Path, f))]


substrates = []
lesion_count = []
latent = []
pretraining = []

for run in runs:
    sub=run.split('-')[1][:-2]
    n = run.split('-')[2].split('_')[0]
    lat = run.split('-')[3].split('_')[0]
    pre = run.split('-')[4]
    substrates.append(sub)
    lesion_count.append(n)
    latent.append(lat)
    pretraining.append(pre)

latent = numpy.unique(latent)
substrates = numpy.unique(substrates)
lesion_count = list(numpy.unique(lesion_count))
pretraining = numpy.unique(pretraining)


DICES = dict()
for sub in substrates:
    DICES[sub] = dict()
    for lat in latent:
        DICES[sub][lat] = dict()
        for pre in pretraining:
            DICES[sub][lat][pre] = numpy.zeros([len(lesion_count),200])

for run in runs:
    sub=run.split('-')[1][:-2]
    n = run.split('-')[2].split('_')[0]
    lat = run.split('-')[3].split('_')[0]
    pre = run.split('-')[4]
    filename = os.path.join(Path, run, 'dice_inference.npy')
    if not os.path.exists(filename):
        continue
    tmp = numpy.load(os.path.join(Path, run, 'dice_inference.npy'))
    DICES[sub][lat][pre][lesion_count.index(n),:] = tmp



for i in range(10):
    plt.subplot(5,2,i+1)
    for sub in substrates:
        plt.plot(DICES[sub][lat][pre][i,:])




import os, numpy, matplotlib.pyplot as plt

Path='/mnt/h/DLDM/VALIDATION'
DICES = numpy.load(os.path.join(Path, 'DICES.npy'), allow_pickle=True).item()

substrates = DICES.keys()
latent = ['True', 'False']
pretraining = ['True', 'False']
lesion_count = ['100','200','300','400','500','600','700','800','900','1000']

# for sub in substrates:
#     tmp = DICES[sub][latent[0]][pretraining[0]][lesion_count.index('100'),:]
#     plt.plot(tmp, label=sub)



substrates_final = []
for sub in substrates:
    tmp = numpy.load(os.path.join(Path, 'substrates',f'{sub}.npy'))
    tmp = numpy.rot90(numpy.sum(tmp, axis = 0),1)
    tmp = numpy.where(tmp>0,1,0)
    print(f'{sub}: {numpy.sum(tmp)}')
    if numpy.sum(tmp) > 10:
        substrates_final.append(sub)


for n in lesion_count:
    plt.subplot(5,2,lesion_count.index(n)+1)
    for sub in substrates_final:
        tmp = DICES[sub][latent[0]][pretraining[0]][lesion_count.index(n),:]
        plt.plot(tmp, label=sub)
    plt.title(f'Lesion Count: {n}')

plt.tight_layout()
plt.suptitle(f'Latent: {latent[0]}, Pretraining: {pretraining[0]}', y=1.02)




