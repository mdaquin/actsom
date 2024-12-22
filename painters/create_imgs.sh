#! /bin/bash

l=`cd painters/base_soms/; ls *.pt; cd -`

for f in $l 
do
   python3 view_som.py painters/base_soms/$f -hl -o painters/imgs/soms/$f.png
done
