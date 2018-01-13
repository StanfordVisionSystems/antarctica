#!/bin/bash

# for reel_num in $(seq 1 57); do
#     echo "processing reel: $reel_num"
#     curdir=$(pwd)
#     echo $curdir
#     pushd .

#     echo cd /mnt/cold/jemmons/geophysics/35mm/Reel_"$reel_num"/TIFF\ \(2400x1800\)/
#     cd /mnt/cold/jemmons/geophysics/35mm/Reel_"$reel_num"/TIFF\ \(2400x1800\)/

#     echo $curdir/antarctica_image_processor --reel "$reel_num" --parallel 48 --flush 1 --output /mnt/data/jemmons/geophysics/finished.updated/Reel_"$reel_num"_output/ * &> /mnt/data/jemmons/geophysics/finished.updated/Reel_"$reel_num"_output.log
#     $curdir/antarctica_image_processor --reel "$reel_num" --parallel 48 --flush 1 --output /mnt/data/jemmons/geophysics/finished.updated/Reel_"$reel_num"_output/ * &> /mnt/data/jemmons/geophysics/finished.updated/Reel_"$reel_num"_output.log

#     popd
# done


for reel_num in $(seq 1 10); do
export REEL=$reel_num && ./antarctica_image_processor --parallel 48 --reel $REEL --flush 1 \
                                                      --output /mnt/data/jemmons/geophysics/finished.new/reel_$REEL \
                                                      /mnt/data/jemmons/geophysics/preprocessed/reel_$REEL/*.tiff &> /mnt/data/jemmons/geophysics/finished.new/reel_$REEL.log
done
