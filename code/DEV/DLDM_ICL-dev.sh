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
sftp -P 22 patrik@144.82.48.21

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


Path="/home/patrik/Data/LDM-ICL"

sudo docker run -it --gpus all -v $Path:/data dldm:dev python

create_training_data.py
create_deficit_data.py





#############################################
#                                           #
#          INITIAL ICL TRAINING             #
#                                           #
#############################################


Path="/home/patrik/Data/LDM-ICL"

sudo docker run -it --gpus all -v $Path:/data -e OUTDIR=out_single-sub_mix-def -e PRETRAINING=True -e ACI=False dldm:dev python

