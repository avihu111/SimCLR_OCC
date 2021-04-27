#!/bin/bash
#SBATCH --mem=10g
#SBATCH -c 6
#SBATCH --time=1-0
#SBATCH --gres=gpu:1,vmem:8g
#SBATCH --array=0-1

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

ARRAY1=(25 50)
ARRAY2=(-)


shape_from_arrays ARRAY1 ARRAY2
ind2sub ${SLURM_ARRAY_TASK_ID} "${shape[@]}"


var1=${ARRAY1[${idxes[0]}]}
var2=${ARRAY2[${idxes[1]}]}

echo ${var1}
echo ${var2}

dir=/cs/labs/daphna/avihu.dekel/simCLR/
cd $dir
source /cs/labs/daphna/avihu.dekel/env/bin/activate

module load torch
python run.py --num-labeled-examples 40 --epochs 50 --workers 6

