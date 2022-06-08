#!/bin/sh
ffmpeg -f concat -i concat.txt -c copy animations.mp4
