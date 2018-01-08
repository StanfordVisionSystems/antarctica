tesseract glacierdigits5.glacierdigits5.exp0.tif glacierdigits5.glacierdigits5.exp0 nobatch box.train

unicharset_extractor glacierdigits5.glacierdigits5.exp0.box

echo "glacierdigtis5 0 0 0 0 0" > font_properties

shapeclustering -F font_properties -U unicharset glacierdigits5.glacierdigits5.exp0.tr

mftraining -F font_properties -U unicharset -O glacierdigits5.unicharset glacierdigits5.glacierdigits5.exp0.tr

cntraining glacierdigits5.glacierdigits5.exp0.tr

mv inttemp glacierdigits5.inttemp
mv normproto glacierdigits5.normproto
mv pffmtable glacierdigits5.pffmtable
mv shapetable glacierdigits5.shapetable
combine_tessdata glacierdigits5.

# copy glacierdigits5.traindata to /usr/share/tesseract-ocr/tessdata/
