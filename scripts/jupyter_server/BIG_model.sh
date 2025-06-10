#!/bin/bash -f

<<<<<<< HEAD

#$ -N BIG_model               # Job name
#$ -o BIG_model2.log           # Output log file name
=======
#$ -N BIG_model              # Job name
#$ -o BIG_model.log           # Output log file name
>>>>>>> ffbde91d0cb45d034a98fb000f93e6a4db6591f0
#$ -l h_rt=168:00:00              # Set wall-clock time limit 
#$ -l vf=64G                     # Memory limit per slot
#$ -cwd                          # Use the current working directory
#$ -j y                          # Join standard error and output logs
#$ -S /bin/bash                  # Use bash shell for the job script
#$ -l lcn_gpu=1                  # Request 1 GPU

# Load CUDA environment
module load apps/nvhpc/24.9/cu11.8/nvhpc
module load libs/cudnn/9.5.1.17/cuda-11

# Activate conda environment
#source /hpc/srs/local/miniconda3/etc/profile.d/conda.sh  # Source conda environment script
#conda activate tf                                        # Activate conda environment
module load apps/miniconda/py311-24.9.2
source /store/software/hpc/apps/miniconda/py311-24.9.2/etc/profile.d/conda.sh
conda activate tf-adam

# Output environment and job information
echo "LIST OF THE MODULES THAT ARE CURRENTLY LOADED:"
module list
echo "--"
echo "CURRENTLY ACTIVE CONDA ENVIRONMENT"
conda env list
echo "--"
echo "Python path: $(which python)"
echo "Working Directory: $(pwd)"
echo "Queue: $QUEUE"
echo "Job ID: $JOB_ID"
echo "Node: $(hostname)"
echo "Number of CPU slots (NSLOTS): ${NSLOTS:-1}"  # Defaults to 1 if not set
echo "Memory requested per slot: 64G"

# Start Jupyter notebook
echo "Starting Python Script..."
python BIG_model.py

exit 0
