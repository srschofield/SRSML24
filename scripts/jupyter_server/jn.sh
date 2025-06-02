#!/bin/bash -f

#$ -q I_40T_64G_NVIDIA_TeslaT4_16G.q -l hostname=m041 
#$ -N jn                         # Job name: 'jn'
#$ -l h_rt=72:00:00              # Set wall-clock time limit 
#$ -l vf=32G                     # Memory limit per slot
#$ -pe ompi-local 1              # Request 16 CPU slots for parallel environment 'ompi-local'
# #$ -M s.schofield@ucl.ac.uk      # Email address for job notifications
# #$ -m bea                        # Email notifications on (b)egin, (e)nd, and (a)bort
#$ -V                            # Export all environment variables to the job
#$ -cwd                          # Use the current working directory
#$ -j y                          # Join standard error and output logs
#$ -o jn_output.log              # Output log file name
#$ -S /bin/bash                  # Use bash shell for the job script
#$ -l lcn_gpu=1                  # Request 1 GPU

# Load CUDA environment
module load apps/nvhpc/24.9/cu11.8/nvhpc
module load libs/cudnn/9.5.1.17/cuda-11

# Activate conda environment
source /hpc/srs/local/miniconda3/etc/profile.d/conda.sh  # Source conda environment script
conda activate tf                                        # Activate conda environment

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
echo "Memory requested per slot: 32G"

# Set up SSH port forwarding back to Apollo
screen -dmS jn ssh -N -L 8888:localhost:8888 srs@apollo.lcn.ucl.ac.uk

# Write server info to file as soon as it's available
jupyter notebook --no-browser --port=8888 --ip=localhost \
  --NotebookApp.allow_origin='*' 2>&1 | tee jn_output.log | grep --line-buffered http > jn_server.log

exit 0
