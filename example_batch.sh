#!/bin/bash
#SBATCH --mem=4g
#SBATCH -c 2
#SBATCH --time=1-0
# #SBATCH --gres=gpu:1
#SBATCH --array=0-17

function ind2sub() {
    local idx="$1"   # Save first argument in a variable
    shift            # Shift all arguments to the left (original $1 gets lost)
    local shape=("$@") # Rebuild the array with rest of arguments
    local cur_idx=$(($idx))  #zero base

    num_dims=${#shape[@]}  # returns the length of an array
    for ((i=0; i<$num_dims; i++))
    do
        cur_dim=${shape[$i]}
        idxes[$i]=$(($cur_idx%$cur_dim))
        # echo ${idxes[$i]}
        # echo $(($cur_idx%$i))
        local cur_idx=$(($cur_idx/$cur_dim))
    done
    # echo $idx
}

function cumprod() {
    local arr=("$@") # Rebuild the array with rest of arguments
    local prod=1
    for ((i=0; i<${#arr[@]}; i++))
    do
        ((prod *= ${arr[$i]}))
    done
    echo $prod
}

function shape_from_arrays() {
    local arr=("$@") # Rebuild the array with rest of arguments
    for ((i=0; i<${#arr[@]}; i++))
    do
        local -n cur_array=${arr[$i]}   # -n for declaring an array
        shape+=(${#cur_array[@]})
    done
}

echo ${SLURM_ARRAY_TASK_ID}

ARRAY1=(1 2)
ARRAY2=(bob moshe gad)
ARRAY3=(0 1 2)
ARRAY4=(-)

shape_from_arrays ARRAY1 ARRAY2 ARRAY3 ARRAY4
ind2sub ${SLURM_ARRAY_TASK_ID} "${shape[@]}"


var1=${ARRAY1[${idxes[0]}]}
var2=${ARRAY2[${idxes[1]}]}
var3=${ARRAY3[${idxes[2]}]}
var4=${ARRAY4[${idxes[3]}]}

echo ${var1}
echo ${var2}
echo ${var3}
echo ${var4}

#dir=/cs/labs/daphna/avihu.dekel/simCLR/
#cd $dir
#source /cs/labs/daphna/avihu.dekel/env/bin/activate

#module load torch

# the SLURM_ARRAY_TASK_ID variable will get the appropiate array index. using the dollar sign is the way to refrence variables in bash (and echo just prints).
#echo ${SLURM_ARRAY_TASK_ID}

#ARRAY=(0.05 0.1 bob)

#yossi=${ARRAY[${SLURM_ARRAY_TASK_ID}]}

#echo ${yossi}

#python run.py --gpu-index 0 --rel-class ${SLURM_ARRAY_TASK_ID} --epochs 1

