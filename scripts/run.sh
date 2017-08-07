#!/bin/bash

for reel_num in $(seq 28 57); do
    echo "processing reel: $reel_num"
    ./antarctica_image_processor --reel "$reel_num" --parallel 48 --flush 1 --output /mnt/data/jemmons/geophysics/finished/Reel_"$reel_num"_output/ /mnt/data/jemmons/geophysics/35mm/Reel_3/TIFF\ \(2400x1800\)/* &> /mnt/data/jemmons/geophysics/finished/Reel_"$reel_num"_output.log
done
