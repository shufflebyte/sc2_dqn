#!/usr/bin/env bash
echo "[START] Make sure pipenv shell is active"
# how to use
# ./start.sh : starts ray server with port 9900, 8 workers, 10 games per worker and tensorboard on port 6006
# ./start.sh 9902 5 7 6009: starts ray server port 9002 with 5 workers and 7 games per worker and tensorboard on 6009


# rayport, standard is 9900
PORT_RAY=$1
if [ -z "$PORT_RAY" ]; then
PORT_RAY=9900
fi

# number of workers (max 8 ports available)
NUMWORKERS=$2
if [ -z "$NUMWORKERS" ]; then
NUMWORKERS=8
fi

# games running parallel per worker
GAMES_PER_WORKER=$3
if [ -z "$GAMES_PER_WORKER" ]; then
GAMES_PER_WORKER=10
fi

# start screen with tensorboard
PORT_TENSORBOARD=$4
if [ -z "$PORT_TENSORBOARD" ]; then
PORT_TENSORBOARD=6006
fi

echo "[START] Starting ray server... #$NUMWORKERS workers"
echo screen -S ray_server -d -m bash -c "CUDA_VISIBLE_DEVICES=1 python ../ray_server.py -n $NUMWORKERS"
# use -L for logging everything
screen -S ray_server -d -m bash -c "CUDA_VISIBLE_DEVICES=1 python ../ray_server.py -n $NUMWORKERS; bash"

sleep 4
echo "[START] Waiting for the server to boot..."
sleep 4
echo "[START] Still waitin..."
sleep 4
echo -e "[START] \xE2\x99\xAB I dont wanna wait... \xE2\x99\xAB"
sleep 4
echo "[START] Sorry, just another 20 seconds..."
sleep 20
echo "[START] Starting $NUMWORKERS workers with $GAMES_PER_WORKER games per worker..."


LOOP=0
# Create 10 workers for each port
while [ "$LOOP" -lt "$GAMES_PER_WORKER" ]; do
    PORT=$((PORT_RAY+1))
    END=$((PORT+NUMWORKERS))
    # Create worker on each port
    while [ "$PORT" -lt "$END" ]; do
        echo screen -S worker_$PORT -d -m bash -c "CUDA_VISIBLE_DEVICES=1 python ../run.py -p $PORT -f;bash"
        screen -S worker_$PORT -d -m bash -c "CUDA_VISIBLE_DEVICES=1 python ../run.py -p $PORT -f; bash"
        let PORT=PORT+1
    done
    let PORT=PORT_RAY+1
    let LOOP=LOOP+1
done
echo "[START] Great Success!"

# start tensorboard automatically
echo screen -S tensorboard_$PORT_TENSORBOARD -d -m bash -c "export LC_ALL=C; conda activate sc2; tensorboard --logdir=~/ray_results --port=$PORT_TENSORBOARD; bash"
screen -S tensorboard_$PORT_TENSORBOARD -d -m bash -c "export LC_ALL=C; conda activate sc2; tensorboard --logdir=~/ray_results --port=$PORT_TENSORBOARD; bash"
