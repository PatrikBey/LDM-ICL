#############################################
#
#
# DLDM_ICL-development
#
#
#
# This script contains the development snippets
# to update the previous deep lesion deficit mapping model
# to incorporate semi-supervised pretraining as well as in-context learning.
#
#
# author: Dr. Patrik Bey. patrik.bey@ucl.ac.uk
#
#
#
#############################################


#############################################
#                                           #
#        CLUSTER CONNECTIONS                #
#                                           #
#############################################
# ---- macbook usage ---- #
ssh-add --apple-use-keychain ~/.ssh/id_ed25519_ucl

ssh patrik@144.82.48.21 -p 22
# sftp -P 22 patrik@144.82.48.21

# ---- neuraxis3 ---- #
ssh pbey@192.168.208.17
# sftp pbey@192.168.208.17


#############################################
#                                           #
#            SET UP CONTAINERS              #
#                                           #
#############################################


Path="/home/patrik/Data/LDM-ICL"

sudo docker run -it --gpus all -v ${Path}:/data dldm:dev python


# # ---- fixing GPU docker runtime issues ---- # 

# # Remove Snap Docker
# sudo snap remove docker

# # Install Docker via apt (native)
# sudo apt update
# sudo apt install apt-transport-https ca-certificates curl software-properties-common

# curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# sudo apt update
# sudo apt install docker-ce docker-ce-cli containerd.io

# # Start and enable Docker
# sudo systemctl enable docker
# sudo systemctl start docker

# # Reinstall NVIDIA Container Toolkit
# sudo apt-get install -y nvidia-container-toolkit
# sudo nvidia-ctk runtime configure --runtime=docker
# sudo systemctl restart docker

# # Test NVIDIA runtime
# docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# # If that works, test your container
# docker run -it --gpus all -v $PWD:/data dldm:dev python


# # ---- Download LDM-ICL repository ---- #
# wget https://www.github.com/PatrikBey/LDM-ICL/archive/refs/heads/main.zip
# 
# unzip main.zip -d /home/patrik/Software/LDM-ICL/Docker



#############################################
#                                           #
#          TRAINING DATA CREATION           #
#                                           #
#############################################


# Path="/home/patrik/Data/LDM-ICL"

# docker run -it --gpus all -v $Path:/data dldm:dev python

# create_training_data.py
# create_deficit_data.py





# #############################################
# #                                           #
# #          INITIAL ICL TRAINING             #
# #                                           #
# #############################################


# Path="/home/patrik/Data/LDM-ICL"

# sudo docker run -it --gpus all -v $Path:/data -e OUTDIR=out_single-sub_mix-def -e PRETRAINING=True -e ACI=False dldm:dev python



# #############################################
# #                                           #
# #          FULL ICL TRAINING             #
# #                                           #
# #############################################

# Path="/home/patrik/Data/LDM-ICL"

# sudo docker run -it --gpus all -v $Path:/data -e OUTDIR=multi-sub_multi-def -e PRETRAINING=True -e ACI=False dldm:dev python


# #############################################
# #                                           #
# #            NEURAXIS RUNS                  #
# #                                           #
# #############################################

# Path="/home/pbey/data"

# docker run -it --gpus device=7 -v $Path:/data -e OUTDIR=test -e PRETRAINING=True -e ACI=False patrikneuro/dldm:dev python


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



# VARIABLES:
SUBSTRATE_TYPE="AAL3_roi_{i}.npy"
N_LESIONS=1000
LATENT_SPLIT="False"
PRETRAINING="True"
DEFICITS_TRAIN="trans"
DEFICITS_TEST="overlap_ratio_noisy"
# DEFAULTS: 
TRAIN_LESION_TYPE="pretrain-tune_5K_2D.npy" 
TEST_LESION_TYPE="predict_5K_2D.npy"


substrate="Schaefer2018_roi_169.npy"
N=200
pt="True"
latent="False"

docker run -it --gpus all -v $PWD:/data -e SUBSTRATE_TYPE="${substrate}" -e N_LESIONS="${N}" -e LATENT_SPLIT=${latent} -e PRETRAINING=${pt} -e DEFICITS_TRAIN="overlap_ratio_noisy" -e DEFICITS_TEST="overlap_ratio_noisy" -e TRAIN_LESION_TYPE="pretrain-tune_5K_2D.npy" -e TEST_LESION_TYPE="predict_5K_2D.npy" -e OUTDIR="out-${substrate%.npy}" dldm:dev python 
/src/call_split/run_1Kx1K.py


docker run -it --gpus all -v $PWD:/data -e TRAIN_LESION_TYPE="pretrain-recon_10K_2D.npy" dldm:dev python 

call_split/run_1Kx1K.py