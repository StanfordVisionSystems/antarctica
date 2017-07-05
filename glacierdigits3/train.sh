# tesseract glacierdigits3.glacierdigits3.exp0.tif glacierdigits3.glacierdigits3.exp0 nobatch box.train

unicharset_extractor glacierdigits3.glacierdigits3.exp0.box

echo "glacierdigtis3 0 0 0 0 0" > font_properties

shapeclustering -F font_properties -U unicharset glacierdigits3.glacierdigits3.exp0.tr

mftraining -F font_properties -U unicharset -O glacierdigits3.unicharset glacierdigits3.glacierdigits3.exp0.tr

cntraining glacierdigits3.glacierdigits3.exp0.tr

mv inttemp glacierdigits3.inttemp
mv normproto glacierdigits3.normproto
mv pffmtable glacierdigits3.pffmtable
mv shapetable glacierdigits3.shapetable
combine_tessdata glacierdigits3.
