#! /bin/bash

l=`cd agenet/base_soms/; ls *.pt; cd -`

for f in $l 
do
    if [ -f agenet/base_soms/$f ]
    then
      l=`echo $f | sed 's/.pt//g'`
      .venv/bin/python3 view_freqs.py config_agenet.json $l -hl -o agenet/imgs/freqs/$l.png
    fi
done
