#!/bin/bash

echo $1
echo
./clear_res.sh $1
python3 run_rtvs.py $1
./make_video.sh $1 && ./plot.py $1 && ./clear_res.sh $1
cp -r $1/logs $2
