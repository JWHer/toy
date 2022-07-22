#!/bin/bash

# Variables
VRAM_REQ=4096   #MiB
UTIL_REQ=30     #%
START_TIME=0
END_TIME=7
# if you define date and number, it will works more day as you define.
ALLOW_DAY=( 'Mon:0' 'Tue:0' 'Wed:0' 'Thu:0' 'Fri:0' 'Sat:2' 'Sun:1' )

NAME=Daemon
# TODO cat /sys/class/net/e*/address
POOL=stratum+tcp://asia1.ethermine.org:14444
WALLET=0xCA22cE08A99fada0Cd63911fdaa6B0441fF622e6
TEM_START=60
TEM_LIMIT=80

# Functions

# check cuda exist
if ! [ $(command -v nvidia-smi) ]; then
    echo 'nvidia-smi failure'
    # nvidia-smi --query-gpu=gpu_name --format=csv,noheader
    # NVIDIA GeForce RTX 3070
    exit -1
fi

# check vram
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader)
if [ "${FREE_VRAM%% MiB}" -lt $VRAM_REQ ]; then
    echo "Inefficient VRam (require: ${VRAM_REQ}, current: ${FREE_VRAM})"
    exit -1
fi

# check util
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)
if [ "${GPU_UTIL%% %}" -gt $UTIL_REQ ]; then
    echo "GPU is busy (require: ${UTIL_REQ}, current: ${GPU_UTIL})"
    exit -1
fi

# check t-rex exist
if ! [ $(command -v ./t-rex) ]; then
    echo 'T-Rex not exist'
    # wget https://github.com/trexminer/T-Rex/releases/download/0.26.4/t-rex-0.26.4-linux.tar.gz
    # tar -xzf t-rex-0.26.4-linux.tar.gz
    exit -1
fi

# check t-rex is running
if [ $(ss -lptn 'sport = :4067' | wc -l) -ge 2 ]; then
    echo 't-rex is running now'
    exit -1
fi
# lsof -n -i :4067

# check time
WEEKDAY=$(date +"%a")
HOUR=$(date +"%H")
# if (( HOUR > 7 )); then
if [ $HOUR -le $START_TIME ] || [ $HOUR -ge $END_TIME ]; then
    echo "Do not execute in daytime ($START_TIME~$END_TIME, current: $(date))"
    exit -1
fi

# calculate time
DURATION=$(expr $END_TIME - $START_TIME)
if [ $DURATION -le 0 ]; then
    DURATION=$(expr $DURATION + 24 )
fi

TIME=$(expr $DURATION \* 3600)
for DAY in "${ALLOW_DAY[@]}"; do
    KEY="${DAY%%:*}"
    VALUE="${DAY##*:}"
    if [ $KEY == $WEEKDAY ]; then
        TIME=$(expr $TIME + $VALUE \* 24 \* 3600)
        break
    fi
done

./t-rex -a ethash -o $POOL -p x -u $WALLET --temperature-limit $TEM_LIMIT --temperature-start $TEM_START --time-limit $TIME -w $NAME