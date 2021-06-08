#!/bin/bash

img="$1"
echo "Processing: ${img}"

stem="${img%.*}"
convert "$img" -gravity center -extent 1:1 -resize 256x256 -grayscale Rec709Luma -format jpeg - \
    | tee "${stem}.jpg" | convert - -depth 8 GRAY:- \
    | raveler -f tsv -N 6000 -k 256 -w 100e-6 -r 256 - | awk '/^[0-9]/{print($1)}' \
    > "${stem}.raveled"
