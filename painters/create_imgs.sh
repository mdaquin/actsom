#! /bin/bash


l=`cd painters/base_soms/; ls *.pt; cd -`

for f in $l 
do
   python3 view_som.py painters/base_soms/$f -hl -o painters/imgs/soms/$f.png
done

python3 view_som.py painters/base_soms/embedding.pt -a mean -hl -o painters/imgs/freqs/embedding.png -d painters/dataset/ -m painters/model.pt -mc painters/model.py -n
python3 view_som.py painters/base_soms/lstm.pt      -a mean -hl -o painters/imgs/freqs/lstm.png      -d painters/dataset/ -m painters/model.pt -mc painters/model.py -n
python3 view_som.py painters/base_soms/hidden.pt    -a mean -hl -o painters/imgs/freqs/hidden.png    -d painters/dataset/ -m painters/model.pt -mc painters/model.py -n
python3 view_som.py painters/base_soms/out.pt       -a mean -hl -o painters/imgs/freqs/out.png       -d painters/dataset/ -m painters/model.pt -mc painters/model.py -n

