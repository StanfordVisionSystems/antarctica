tesseract glacierdigits4.glacierdigits4.exp0.tif glacierdigits4.glacierdigits4.exp0 nobatch box.train

unicharset_extractor glacierdigits4.glacierdigits4.exp0.box

echo "glacierdigtis4 0 0 0 0 0" > font_properties

shapeclustering -F font_properties -U unicharset glacierdigits4.glacierdigits4.exp0.tr

mftraining -F font_properties -U unicharset -O glacierdigits4.unicharset glacierdigits4.glacierdigits4.exp0.tr

cntraining glacierdigits4.glacierdigits4.exp0.tr

mv inttemp glacierdigits4.inttemp
mv normproto glacierdigits4.normproto
mv pffmtable glacierdigits4.pffmtable
mv shapetable glacierdigits4.shapetable
combine_tessdata glacierdigits4.

# copy glacierdigits4.traindata to /usr/share/tesseract-ocr/tessdata/
