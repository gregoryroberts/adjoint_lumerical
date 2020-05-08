#!/bin/bash

#SBATCH -A Faraon_Computing
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --qos=normal
#SBATCH --mem=16G

echo "LOG" > slurms.out

NUM_WORKERS=3

worker_slurm_ids=()

for WORKER_ID in $(seq 1 $NUM_WORKERS)
do	
	SLURM_ID=$(sbatch launch_worker.sh $WORKER_ID | tr -dc '0-9')
	worker_slurm_ids+=( $SLURM_ID )
done

echo ${worker_slurm_ids[*]} >> slurms.out

while true; do
	NUM_WORKERS_STARTED=0
	declare -i NUM_WORKERS_STARTED

	for WORKER_ID in $(seq 1 $NUM_WORKERS)
	do
		#echo $(squeue -u gdrobert --state=running) >> slurms.out
		echo ${worker_slurm_ids[$WORKER_ID]} >> slurms.out
		echo $(squeue -u gdrobert --state=running | grep ${worker_slurm_ids[$WORKER_ID]}) >> slurms.out
		if squeue -u gdrobert --state=running | grep ${worker_slurm_ids[$WORKER_ID]}
		then
			NUM_WORKERS_STARTED+=1
		fi
	done

	if [ $NUM_WORKERS_STARTED == $NUM_WORKERS ]
	then
		break
	fi

	echo $NUM_WORKERS_STARTED >> slurms.out

	sleep 5
done


exit $?
