tesseract glacierdigits7.glacierdigits7.exp0.tif glacierdigits7.glacierdigits7.exp0 nobatch box.train

unicharset_extractor glacierdigits7.glacierdigits7.exp0.box

echo "glacierdigtis7 0 0 0 0 0" > font_properties

shapeclustering -F font_properties -U unicharset glacierdigits7.glacierdigits7.exp0.tr

mftraining -F font_properties -U unicharset -O glacierdigits7.unicharset glacierdigits7.glacierdigits7.exp0.tr

cntraining glacierdigits7.glacierdigits7.exp0.tr

mv inttemp glacierdigits7.inttemp
mv normproto glacierdigits7.normproto
mv pffmtable glacierdigits7.pffmtable
mv shapetable glacierdigits7.shapetable
combine_tessdata glacierdigits7.

# copy glacierdigits7.traindata to /usr/share/tesseract-ocr/tessdata/
