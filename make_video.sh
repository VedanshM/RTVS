#!/bin/bash
ffmpeg -f image2  -framerate 250 -i $1/results/test.rgba.00000.%05d.png -c:v libx264 -r 24 -pix_fmt yuv420p  -y  $1/logs/out.mp4
ln -sf $1/logs/out.mp4 out.mp4
