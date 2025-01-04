#! /bin/bash

l=`cd painters/base_soms/; ls *.pt; cd -`

for f in $l 
do
    if [ -f painters/base_soms/$f ]
    then
      .venv/bin/python3 view_freqs.py config_painters.json -hl -o painters/freqs/soms/$l.png
    fi
done
