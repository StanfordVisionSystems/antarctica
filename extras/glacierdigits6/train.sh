tesseract glacierdigits6.glacierdigits6.exp0.tif glacierdigits6.glacierdigits6.exp0 nobatch box.train

unicharset_extractor glacierdigits6.glacierdigits6.exp0.box

echo "glacierdigtis6 0 0 0 0 0" > font_properties

shapeclustering -F font_properties -U unicharset glacierdigits6.glacierdigits6.exp0.tr

mftraining -F font_properties -U unicharset -O glacierdigits6.unicharset glacierdigits6.glacierdigits6.exp0.tr

cntraining glacierdigits6.glacierdigits6.exp0.tr

mv inttemp glacierdigits6.inttemp
mv normproto glacierdigits6.normproto
mv pffmtable glacierdigits6.pffmtable
mv shapetable glacierdigits6.shapetable
combine_tessdata glacierdigits6.

# copy glacierdigits6.traindata to /usr/share/tesseract-ocr/tessdata/
