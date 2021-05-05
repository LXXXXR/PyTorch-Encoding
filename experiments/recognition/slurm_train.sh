#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:5}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u experiments/recognition/train_dist_modified.py --dataset imagenet --model resnest50 --lr-scheduler cos --epochs 270 --checkname resnest50 --lr 0.025 --batch-size 64 --dist-url tcp://MASTER:NODE:IP:ADDRESS:23456 --world-size 2 --label-smoothing 0.1 --mixup 0.2 --no-bn-wd --last-gamma --warmup-epochs 5 --rand-aug 