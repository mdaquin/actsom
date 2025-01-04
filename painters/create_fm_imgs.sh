#! /bin/bash

l=`cd painters/base_soms/; ls *.pt; cd -`

for f in $l 
do
    if [ -f painters/base_soms/$f ]
    then
      l=`echo $f | cut -d'.' -f1`
      .venv/bin/python3 view_freqs.py config_painters.json $l -hl -o painters/imgs/freqs/$l.png
    fi
done
