#!/bin/sh
ffmpeg -y -f concat -i concat.txt -c copy animations.mp4
