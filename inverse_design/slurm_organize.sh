#!/bin/bash

#SBATCH -A Faraon_Computing
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --mem=16G

NUM_WORKERS=3

worker_slurm_ids=()

for WORKER_ID in { 1 .. $NUM_WORKERS }
do	
	SLURM_ID=$(sbatch launch_worker.sh $WORKER_ID | tr -dc '0-9')
	worker_slurm_ids+=( $SLURM_ID )
done

echo $worker_slurm_ids > slurms.out

exit $?
