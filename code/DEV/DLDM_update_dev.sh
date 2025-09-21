#############################################
#
#
# DLDM_update_dev.sh
#
#
#
# This script contains the development snippets
# to update the previous deep lesion deficit mapping model
# to incorporate patient covariants.
#
#
# author: Dr. Patrik Bey. patrik.bey@ucl.ac.uk
#
#
#


# # converting torch.DoubleTensor to torch.masked
# from torch.mask import masked_tensor

# data = torch.arange(12, dtype=torch.float).reshape(3, 4)
# mask = torch.tensor([[True, False, False, True], [False, True, False, False], [True, True, True, True]])
# mt = torch.masked_tensor(data, mask)
#     >>> data.select(0, 1)
#     tensor([4., 5., 6., 7.])
#     >>> mask.select(0, 1)
#     tensor([False,  True, False, False])
#     >>> mt.select(0, 1)
#     MaskedTensor(
#       [      --,   5.0000,       --,       --]
#     )


#############################################
#                                           #
#           PRETRAINING SPLIT               #
#                                           #
#############################################

docker build . -t dldm:dev -f Docker/Dockerfile
$Date = $(Get-Date -f dd-MM-yyyy)
mkdir "H:\DLDM\TESTING\$Date"
clear
# docker run --gpus=all -v H:\DLDM\TESTING\"$Date":/data dldm:dev python /src/ldm_pretrain_vae/run.py

docker run -it --gpus=all -v H:\DLDM\TESTING\"$Date":/data dldm:dev python

import os
# os.chdir('recon_vae')
os.chdir('ldm_raw_vae')

# /src/cov_base_model_run.py


docker run -it --gpus=all -v H:\DLDM\TESTING\"$Date":/data -e N_LESIONS=100 -e OUTDIR=output_100 -e PRETRAINING=True -e SUBSTRATE_TYPE=two_point_substrate_2D.npy -e DEFICIT_TYPE=overlap_ratio dldm:dev python 

Path=/mnt/h/DLDM/TESTING/04-09-2025
END=500
substrates="two_point_substrate_2D.npy question_mark_substrate_2D.npy circular_substrate_2D.npy"
trainings="True False"
ACI="True False"
for aci in ${ACI}; do
    for gt in $substrates; do
        for pt in $trainings; do
            for i in $(seq 25 25 $END); do
                n_lesions=${i}
                for run in $(seq 1 10); do
                    docker run --gpus=all -v $Path:/data -e N_LESIONS=${n_lesions} -e OUTDIR=out_N-${n_lesions}_aci-${aci}_pre-${pt}_gt-${gt}_run-${run} -e PRETRAINING=${pt} -e ACI=${aci} -e SUBSTRATE_TYPE=${gt} -e DEFICIT_TYPE=overlap_ratio dldm:dev python /src/2D/call/run.py
                done
            done
        done
    done
done



Path=/mnt/h/DLDM/TESTING
docker run -it --gpus=all -v $Path:/data -e N_LESIONS=${n_lesions} -e OUTDIR=binary_overlay_pt_true -e PRETRAINING=${pt} -e ACI=${aci} -e SUBSTRATE_TYPE=${gt} dldm:dev python /src/2D/call/run.py


# ---- checking noise impact ---- #
Path=/mnt/h/DLDM/TESTING
trainings="True False"
substrates="two_point_substrate_2D.npy question_mark_substrate_2D.npy circular_substrate_2D.npy"
noises=$(seq 0.1 0.1 1)

for pt in ${trainings}; do
    for gt in $substrates; do
        for noise in $noises; do
            docker run -it --gpus=all -v ${Path}/10-09-2025:/data -e OUTDIR=out-${pt}_gt-${gt%.npy}_noise-${noise}-2 -e PRETRAINING=${pt} -e ACI=False -e SUBSTRATE_TYPE=${gt} -e DEFICIT_TYPE=overlap_ratio_noisy -e DEFICIT_NOISE=${noise} dldm:dev python /src/2D/call/run.py
        done
    done
done

gt="two_point_substrate_2D.npy"
pf=False
docker run -it --gpus=all -v ${Path}/10-09-2025:/data -e OUTDIR=out-${pt}_gt-${gt%.npy}_noise-None -e PRETRAINING=${pt} -e ACI=False -e SUBSTRATE_TYPE=${gt} -e DEFICIT_TYPE=overlap_ratio dldm:dev python /src/2D/call/run.py


noise=0.5
gt="two_point_substrate_2D.npy"
pt=True
docker run -it --gpus=all -v ${Path}/10-09-2025:/data -e OUTDIR=out-${pt}_gt-${gt%.npy}_noise-${noise}-2 -e PRETRAINING=${pt} -e ACI=False -e SUBSTRATE_TYPE=${gt} -e DEFICIT_TYPE=overlap_ratio_noise -e DEFICIT_NOISE=${noise} dldm:dev python /src/2D/call/run.py

noise=0.1
gt="two_point_substrate_2D.npy"
pt=True
docker run -it --gpus=all -v ${Path}/10-09-2025:/data -e OUTDIR=out-${pt}_gt-${gt%.npy}_noise-${noise}-2 -e PRETRAINING=${pt} -e ACI=False -e SUBSTRATE_TYPE=${gt} -e DEFICIT_TYPE=overlap_ratio_noise -e DEFICIT_NOISE=${noise} dldm:dev python /src/2D/call/run.py


gt="two_point_substrate_2D.npy"
trainings="True False"
for pt in ${trainings}; do
    docker run -it --gpus=all -v ${Path}/10-09-2025:/data -e OUTDIR=out-${pt}_gt-${gt%.npy}_distance -e PRETRAINING=${pt} -e ACI=False -e SUBSTRATE_TYPE=${gt} -e DEFICIT_TYPE=distance dldm:dev python /src/2D/call/run.py
done









##############################################
#                                            #
#           RERUN 17-09-2025             #
#                                            #
##############################################

Path=/mnt/h/DLDM/TESTING/17-09-2025

pt=True
gt="cognition_substrate_2D.npy"
latent=True
N=1000
docker run -it --gpus=all -v ${Path}:/data -e PRETRAINING=${pt} -e ACI=False -e SUBSTRATE_TYPE=${gt} -e DEFICIT_TYPE=overlap_ratio -e N_LESIONS=${N} -e LATENT_SPLIT=${latent} -e OUTDIR=out-${pt}_gt-${gt%.npy}_latent-${latent}_n-${N} -e Z_DIM=40 dldm:dev python /src/2D/call_split/run.py



LESIONCOUNT=$(seq 200 100 1000)
TRAININGS="True False"
# SUBSTRATES="two_point_substrate_2D.npy cognition_substrate_2D.npy motor_substrate_2D.npy"
# SUBSTRATES="two_point_substrate_2D.npy cognition_substrate_2D.npy motor_substrate_2D.npy"
SUBSTRATES="cognition_substrate_2D.npy"
LATENTS="True False"

for N in ${LESIONCOUNT}; do
    for pt in ${TRAININGS}; do
        for gt in ${SUBSTRATES}; do
            for latent in ${LATENTS}; do
                if [ ${latent} = "True" ]; then
                    ZDIM=40
                else
                    ZDIM=20
                fi
                docker run -it --gpus=all -v ${Path}:/data -e PRETRAINING=${pt} -e ACI=False -e SUBSTRATE_TYPE=${gt} -e DEFICIT_TYPE=overlap_ratio_noisy -e N_LESIONS=${N} -e LATENT_SPLIT=${latent} -e OUTDIR=Frozen/out-${pt}_gt-${gt%.npy}_latent-${latent}_n-${N} -e Z_DIM=${ZDIM} -e DEFICIT_NOISE=0.25 dldm:dev python /src/2D/call_split/run.py
            done
        done
    done
done


# ---- freezing weights ---- #

# Path=/mnt/h/DLDM/TESTING/17-09-2025

# pt=True
# gt="cognition_substrate_2D.npy"
# latent=True
# N=1000
docker run -it --gpus=all -v ${Path}:/data -e PRETRAINING=${pt} -e ACI=False -e SUBSTRATE_TYPE=${gt} -e DEFICIT_TYPE=overlap_ratio -e N_LESIONS=${N} -e LATENT_SPLIT=${latent} -e OUTDIR=Frozen/out-${pt}_gt-${gt%.npy}_latent-${latent}_n-${N} -e Z_DIM=40 dldm:dev python



##############################################
#                                            #
#             RERUN 19-09-2025               #
#                                            #
##############################################

Path=/mnt/h/DLDM/TESTING/19-09-2025

pt=True
gt="cognition_substrate_2D.npy"
latent=True
N=1000
docker run -it --gpus=all -v ${Path}:/data dldm:dev python
