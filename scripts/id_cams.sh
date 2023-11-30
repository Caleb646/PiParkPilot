#!/bin/bash
set -e
# Usage ./id_cams.sh 0 or ./id_cams.sh 1
# figure out which webcam is which.
fswebcam id_img.jpg --device /dev/video$0