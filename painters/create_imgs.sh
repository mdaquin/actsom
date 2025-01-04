#! /bin/bash

l=`cd painters/base_soms/; ls *.pt; cd -`

for f in $l 
do
    if [ -f painters/base_soms/$f ]
    then
      .venv/bin/python3 view_som.py painters/base_soms/$f -hl -o painters/imgs/soms/$f.png
    fi
done
