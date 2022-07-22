#!/bin/bash

# Variables
VRAM_REQ=4096   #MiB
UTIL_REQ=30     #%
START_TIME=0
END_TIME=7
# if you define date and number, it will works more day as you define.
ALLOW_DAY=(
    'Mon:0' 'Tue:0' 'Wed:0' 'Thu:0' 'Fri:2' 'Sat:2' 'Sun:1'
    '월:0' '화:0' '수:0' '목:0' '금:3' '토:2' '일:1'
)

NAME=Daemon
# TODO cat /sys/class/net/e*/address
POOL=stratum+tcp://asia1.ethermine.org:14444
WALLET=0xCA22cE08A99fada0Cd63911fdaa6B0441fF622e6
TEM_START=60
TEM_LIMIT=80

#################### type test codes ####################
# calculate time
DURATION=$(expr $END_TIME - $START_TIME)
if [ $DURATION -le 0 ]; then
    DURATION=$(expr $DURATION + 24 )
fi
TIME=$(expr $DURATION \* 3600)
WEEKDAY=$(date +"%a")
for DAY in "${ALLOW_DAY[@]}"; do
    KEY="${DAY%%:*}"
    VALUE="${DAY##*:}"
    echo $WEEKDAY $KEY $VALUE
    if [ $KEY == $WEEKDAY ]; then
        TIME=$(expr $TIME + $VALUE \* 24 \* 3600)
        break
    fi
done
echo $TIME
#################### end of test codes ####################

echo 'OK'