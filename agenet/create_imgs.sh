#! /bin/bash

l=`cd agenet/base_soms/; ls *.pt; cd -`

for f in $l 
do
    if [ -f agenet/base_soms/$f ]
    then
      .venv/bin/python3 view_som.py agenet/base_soms/$f -hl -o agenet/imgs/soms/$f.png
    fi
done
