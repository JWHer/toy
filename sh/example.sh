#!/bin/bash

# check parameters
if [ $# -ne 6 ]; then
    echo Parameter error!
	echo Usage $0 name model.json data.json model.pth git_uri branch
    echo '(All paths should be absolute, and under the current $HOME dir)'
	exit -1
else
	echo Got:
    echo $@
    echo ""
fi

# check prerequisite
if ! [ $(command -v jq) ]; then
	echo jq is required
	echo 'please enter "apt install jq" to install it'
	exit -1
fi

function remove_docker {
    docker-compose down
    sudo rm -rf $MOUNT_STORAGE_ROOT
}

# check compose is running
if [ $(docker-compose ps | wc -l) -eq 10 ]; then
    echo containers already running
    exit -1
fi

docker-compose --env-file=./.env up -d

if [ $? -ne 0 ]; then
	echo TX creation faild
	echo 'This error caused by docker-compose (are you miss somthing?)'
    echo ""
	remove_docker
    exit -1
fi

# declare variables
export $(grep -v '^#' .env | xargs)
STATUS=0
COUNT=0
STOP=30 # x3 SECONDS
ENDPOINT=http://localhost:$PORT/api/v1

# health check
while [ $STATUS -ne 200 ]; do
	if [ $COUNT -eq $STOP ]; then
		echo Network failure
		echo Cannot connect to TX
		remove_docker
        exit -1
	fi
        # echo request to $ENDPOINT/project/
        echo -n .
        
        COUNT=$(($COUNT+1))
        STATUS=$(curl --silent -o /dev/null -w "%{http_code}" $ENDPOINT/project/)
        sleep 1
done

echo TX is now ready

# Make new production id
EXP_ID=0
PROD_ID=$(cat /proc/sys/kernel/random/uuid)
# echo $PROD_ID

# create model
MODEL=$(curl --silent -X 'POST' \
  "$ENDPOINT/model/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
  \"name\": \"$1\",
  \"classes\": $(cat $2 | jq .classes),
  \"desc\": \"test model for $1\",
  \"location\": \"$ENDPOINT/storage/model/$PROD_ID\",
  \"capacity\": 1,
  \"version\": \"$(echo $1)_1.0-${PROD_ID:(-8)}_compute86\",
  \"platform\": \"dgpu\",
  \"framework\": \"deepstream==6.0\",
  \"precision\": \"FP16\",
  \"production_id\": \"$PROD_ID\"
}")
# echo $MODEL
RUN_ID=$(echo $MODEL | jq .id)
RUN_ID=$(echo $RUN_ID | tr -d '"')

echo ""
echo Your model $1 id is $RUN_ID !
echo ""

# copy model files
MODEL_DIR=$MOUNT_STORAGE_ROOT/$STORAGE_DIR/$MLFLOW_DIR/$EXP_ID/$RUN_ID/artifacts/model/data
sudo mkdir -p $MODEL_DIR
sudo cp $2 $MODEL_DIR/model.json
sudo cp $3 $MODEL_DIR/data.json
sudo cp $4 $MODEL_DIR/model.pth

# create train
TRAIN=$(curl --silent -X 'POST' \
  "$ENDPOINT/train/" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d "{
  \"name\": \"$1\",
  \"tags\": {\"parent_run_id\":\"$RUN_ID\"},
  \"uri\": \"$5\",
  \"entry_point\": \"tools/train.py\",
  \"version\": \"$6\",
  \"parameters\": {
    \"exp_name\": \"$1\",
    \"model_cfg\": \"$2\",
    \"data_cfg\": \"$3\",
    \"ckpt\": \"$4\"
  },
  \"experiment_id\": \"$EXP_ID\",
  \"use_conda\": false,
  \"synchronous\": false
}")
TRAIN_ID=$(echo $TRAIN | jq .id)
TRAIN_ID=$(echo $TRAIN_ID | tr -d '"')

echo Your train $1 id is $TRAIN_ID !
echo ""
echo Here is the link to check out your train
echo http://$EXTERNAL_DOMAIN:$MLFLOW_PORT/\#/experiments/$EXP_ID
echo ""

# check status
STATUS='"RUNNING"'
while [ $STATUS == '"RUNNING"' ]; do
	STATUS=$(curl --silent -X 'GET' \
    "$ENDPOINT/train/$TRAIN_ID" \
    -H 'accept: application/json' | jq .status)
    # echo $STATUS
	echo -n .
	sleep 3
done

echo ""
echo "train end! (status: $STATUS)"
echo ""

if ! [ $STATUS == '"FINISHED"' ]; then
    docker logs autocare_docker_tx_service >> $1\_failed_log.txt
    echo Failed log saved
fi

echo -n 'Would you like to show logs? (Yes[default]/no): '
read SELECT
if ! [ "$SELECT" = "no" ]; then
    docker logs autocare_docker_tx_service | less
fi

echo ""
echo -n 'Would you like to remove TX? (Yes[default]/no): '
read SELECT
if ! [ "$SELECT" = "no" ]; then
    echo ""
    remove_docker
fi

exit 0
