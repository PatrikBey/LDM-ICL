#############################################
#
#
# DLDM_3D_testing.sh
#
#
#
# This script contains the development snippets
# to update the previous deep lesion deficit mapping model
#
#
# author: Dr. Patrik Bey. patrik.bey@ucl.ac.uk
#
# date: 2025-10-14
#

Path="/mnt/h/DLDM/3D"

docker run -it --gpus=all -v ${Path}:/data dldm:dev python
