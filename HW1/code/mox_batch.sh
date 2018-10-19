#!/bin/bash

## Job Name

#SBATCH --job-name=mnist_ridge

## Allocation Definition
## Which queue do you want to use?

#SBATCH --partition=stf

## Number of simulations that are being done. (Max number is 28?)
## By default 1 CPU core per task is used. Can be changed with --ntasks

#SBATCH --tasks=16

## How many nodes are you using?

#SBATCH --nodes=1

## Walltime needed (Days-Hours:Minutes:Seconds)

#SBATCH --time=0-3:00:00

## Memory per node

#SBATCH --mem=100G

## Email beginning and end of job

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kowash@uw.edu

## Specify the working directory for this job (Where is job located?)

#SBATCH --workdir=/gscratch/home/kowash/ml_class/HW1/code

##START WORK SECTION



START=$(date +%s.%N)
module load anaconda3_4.3.1
mapfile -t indices < /gscratch/home/kowash/ml_class/HW1/code/index_list
for ((i=0; i < ${#indices[@]}; i+=8)); do
    for p in "${indices[@]:$i:8}"; do
        python mnist_ridge.py $p &
    done
    wait
done
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $DIFF
