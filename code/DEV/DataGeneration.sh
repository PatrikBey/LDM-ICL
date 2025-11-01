############################################
#                                          #
#            DATA PROCESSING               #
#                                          #
############################################
#
#
#
# This script performs generation and preprocessing of 
# synthetic lesion and behavioral data for validation experiments.
#
# Created synthetic prior includes:
#
# - 10K lesion masks for reconstruction pretraining
# - 5K lesion masks for finetuning
# - 5K lesion masks for validation
# - prepare substrates based on AAL3 & Schaefer parcellations
# - behavioral data for two deficits (noisy overlap, transformation) for
#   wide ranging substrates

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


# ---- use docker image ---- #
docker run -it -v /home/pbey/data:/data patrikneuro/dldm:dev python
