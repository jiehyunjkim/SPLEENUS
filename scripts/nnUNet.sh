#!/bin/bash

# Sample slurm submission script for the Chimera compute cluster
# Lines beginning with # are comments, and will be ignored by
# the interpreter.  Lines beginning with #SBATCH are directives
# to the scheduler.  These in turn can be commented out by
# adding a second # (e.g. ##SBATCH lines will not be processed
# by the scheduler).
#
#
# set name of job
#SBATCH --job-name=spleenex_nnUNet
#
# set the number of processors/tasks needed
#SBATCH -n 24

# set max wallclock time  DD-HH:MM:SS

# the default time will be 1 hour if not set
##SBATCH --time=01-1:00:00
#SBATCH --time=01-0:00:00

# set a memory request
##SBATCH --mem=512gb
##SBATCH --mem=256gb
#SBATCH --mem=128gb

# Set filenames for stdout and stderr.  %j can be used for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=LogFiles/%x-%j.err
#SBATCH --output=Output/%x-%j.out
#

# set the partition where the job will run.  Multiple partitions can
# be specified as a comma separated list
# Use command "sinfo" to get the list of partitions
#SBATCH --account=aicore
#SBATCH --partition=AICORE_A100
#SBATCH --gres=gpu:A100:1
#SBATCH -w chimera13

#When submitting to the GPU node, these following three lines are needed:

##SBATCH --gres=gpu:1
##SBATCH --export=NONE
#source /etc/profile
##SBATCH --gres=gpu:1

#Optional
# mail alert at start, end and/or failure of execution
# see the sbatch man page for other options
##SBATCH --mail-type=ALL
# send mail to this address
#SBATCH --mail-user=JieHyun.Kim001@umb.edu
# Put your job commands here, including loading any needed
# modules or diagnostic echos. 

# this job simply reports the hostname and sleeps for two minutes

eval "$(conda shell.bash hook)"
conda activate impact-team-2

python3 /home/jiehyun.kim001/inia/scripts/run_nnunet.py

conda deactivate
