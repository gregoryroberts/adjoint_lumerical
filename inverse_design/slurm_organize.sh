#!/bin/bash

#SBATCH -A Faraon_Computing
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --mem=16G

echo "LOG" > slurms.out

NUM_WORKERS=5
NUM_WORKERS_MINUS_ONE=4

worker_slurm_ids=()

for WORKER_ID in $(seq 0 $NUM_WORKERS_MINUS_ONE)
do	
	SLURM_ID=$(sbatch launch_worker.sh /central/groups/Faraon_Computing/projects/layered_infrared_3layers_pol_insensitive_thicker_layers_and_spacers_6x6x4p32um_f4_v3_parallel/ $WORKER_ID | tr -dc '0-9')
	worker_slurm_ids+=( $SLURM_ID )
done

echo ${worker_slurm_ids[*]} >> slurms.out

while true; do
	NUM_WORKERS_STARTED=0
	declare -i NUM_WORKERS_STARTED

	for WORKER_ID in $(seq 0 $NUM_WORKERS_MINUS_ONE)
	do
		if squeue -u gdrobert --state=running | grep ${worker_slurm_ids[$WORKER_ID]}
		then
			NUM_WORKERS_STARTED+=1
		fi
	done

	echo $NUM_WORKERS_STARTED >> slurms.out


	if [ $NUM_WORKERS_STARTED == $NUM_WORKERS ]
	then
		break
	fi


	sleep 5
done

source activate fdtd

xvfb-run --server-args="-screen 0 1280x1024x24" python LayeredLithographyIROptimizationParallel.py $NUM_WORKERS > stdout_ir_parallel.log 2> stderr_ir_parallel.log

exit $?
