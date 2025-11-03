############################################
#                                          #
#          VALIDATION EXPERIMENTS          #
#                                          #
############################################
#
#
#
# This script runs a series of validation experiments for the project.
#
#
# Each experimental setup is run multiple times. Runs performed are defined by:
# - reconstruction pre-training
# - N-lesions finetuning (N={200,100,1000})
# - latent space | shared vs split
#
#
# 1. SINGLE DEFICIT
# 1.1 noisy overlap prediction
# 1.2 transformation prediction
# 2. CROSS DEFICIT
# 2.1 noisy overlap training | transformation prediction
# 2.2 transformation training | noisy overlap prediction
#
# SUBSTRATES
# - Schaefer2018: left hemisphere ROIs N=200 (asymmetric)
# - AAL3: left hemisphere ROIs N=83 (asymmetric)

# ---- PREPARE CLUSTER CONNECTIVITY ---- #

# ---- bolzano connection ---- #
ssh patrik@144.82.48.21 -p 22
sftp -P 22 patrik@144.82.48.21


# # # ---- neuraxis3 connection ---- #
# ssh pbey@192.168.208.17
# sftp pbey@192.168.208.17

# # ---- neuraxis4 connection ---- #
ssh pbey@192.168.208.18
sftp pbey@192.168.208.18


# ---- RECONSTRUCTION PRETRAINING ---- #
Path="/home/pbey/data"
# # Path="/home/patrik/Data/LDM"
# Path="/mnt/h/DLDM/VALIDATION/LDM"

docker run -it --gpus device=0 -v $PWD:/data patrikneuro/dldm:dev python


# ---- INFERENCE FINETUNING ---- #


# ---- initial testing
# Path="/home/pbey/data"

# docker run -it --gpus device=7 -v $Path:/data patrikneuro/dldm:dev python






Path="/home/pbey/data"
SUBSTRATES=$(ls $Path/substrates)
LATENTS="True False"
LESIONCOUNTS=$( seq 100 100 1000)
TRAINING="True False"

mkdir -p $Path/logs
mkdir -p $Path/outputs

# ---- with pretraining ---- #
pt="True"
for substrate in $SUBSTRATES; do
    for latent in $LATENTS; do
        for n in $LESIONCOUNTS; do
            OUTDIR="out-${substrate%.npy}_n-${n}_lat-${latent}_pt-${pt}"
            touch $Path/logs/${OUTDIR}.log
            docker run -it --gpus device=5 -v $Path:/data -e SUBSTRATE_TYPE="${substrate}" -e N_LESIONS="${n}" -e LATENT_SPLIT=${latent} -e PRETRAINING=${pt} -e DEFICITS_TRAIN="overlap_ratio_noisy" -e DEFICITS_TEST="overlap_ratio_noisy" -e TRAIN_LESION_TYPE="pretrain-tune_5K_2D.npy" -e TEST_LESION_TYPE="predict_5K_2D.npy" -e OUTDIR="outputs/${OUTDIR}" patrikneuro/dldm:dev python /src/call_split/run_1Kx1K.py >> $Path/logs/${OUTDIR}.log 2>&1
        done
    done
done

# --- without pretraining ---- #
pt="False"
for substrate in $SUBSTRATES; do
    for latent in $LATENTS; do
        for n in $LESIONCOUNTS; do
            OUTDIR="out-${substrate%.npy}_n-${n}_lat-${latent}_pt-${pt}"
            touch $Path/logs/${OUTDIR}.log
            docker run -it --gpus device=4 -v $Path:/data -e SUBSTRATE_TYPE="${substrate}" -e N_LESIONS="${n}" -e LATENT_SPLIT=${latent} -e PRETRAINING=${pt} -e DEFICITS_TRAIN="overlap_ratio_noisy" -e DEFICITS_TEST="overlap_ratio_noisy" -e TRAIN_LESION_TYPE="pretrain-tune_5K_2D.npy" -e TEST_LESION_TYPE="predict_5K_2D.npy" -e OUTDIR="${OUTDIR}" patrikneuro/dldm:dev python /src/call_split/run_1Kx1K.py >> $Path/logs/${OUTDIR}.log 2>&1
        done
    done
done




# ---- test weight freezing ---- #


Path="/mnt/h/DLDM/VALIDATION/LDM"

docker run -it --gpus all -v $Path:/data -e SUBSTRATE_TYPE="AAL3_roi_1.npy" -e N_LESIONS="1000" -e LATENT_SPLIT=False -e PRETRAINING=True -e DEFICITS_TRAIN="overlap_ratio_noisy" -e DEFICITS_TEST="overlap_ratio_noisy" -e TRAIN_LESION_TYPE="pretrain-tune_5K_2D.npy" -e TEST_LESION_TYPE="predict_5K_2D.npy" patrikneuro/dldm:dev python 





##################################
#                                #
#           3D REWORK            #
#                                #
##################################

# ---- bolzano connection ---- #
# ssh patrik@144.82.48.21 -p 22
# sftp -P 22 patrik@144.82.48.21


# # # ---- neuraxis3 connection ---- #
# ssh pbey@192.168.208.17
# sftp pbey@192.168.208.17




# ---- RECONSTRUCTION PRETRAINING ---- #
Path="/home/pbey/data"
# # Path="/home/patrik/Data/LDM"
# Path="/mnt/h/DLDM/VALIDATION/LDM"

docker run -it --gpus device=5 -v $Path:/data -e N_LESIONS=1000 -e LATENT_SPLIT=False -e PRETRAINING=True -e DEFICITS_TRAIN="overlap_ratio_noisy" -e DEFICITS_TEST="overlap_ratio_noisy" -e TRAIN_LESION_TYPE="pretrain-recon_10K.npy" -e TEST_LESION_TYPE="predict_5K.npy" patrikneuro/dldm:dev python



# ---- INFERENCE FINETUNING ---- #