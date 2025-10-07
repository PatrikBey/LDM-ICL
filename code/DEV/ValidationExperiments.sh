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
# sftp -P 22 patrik@144.82.48.21


# # ---- neuraxis3 connection ---- #
ssh pbey@192.168.208.17
# sftp pbey@192.168.208.17




# ---- RECONSTRUCTION PRETRAINING ---- #
Path="/home/pbey/data"
# Path="/home/patrik/Data/LDM"
# Path="/mnt/h/DLDM/VALIDATION/LDM"

docker run -it --gpus device=7 -v $Path:/data patrikneuro/dldm:dev python



# ---- INFERENCE FINETUNING ---- #


# ---- initial testing
Path="/home/pbey/data"

docker run -it --gpus device=7 -v $Path:/data patrikneuro/dldm:dev python
