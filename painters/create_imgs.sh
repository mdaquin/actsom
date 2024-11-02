hm# /bin/bash

python3 view_som.py painters/base_soms/embedding.pt -hl -o painters/imgs/soms/embedding.png
python3 view_som.py painters/base_soms/lstm.pt      -hl -o painters/imgs/soms/lstm.png
python3 view_som.py painters/base_soms/hidden.pt    -hl -o painters/imgs/soms/hidden.png

python3 view_som.py painters/base_soms/embedding.pt -hl -o painters/imgs/freqs/embedding.png -d painters/dataset/ -m painters/model.pt -mc painters/model.py -n
python3 view_som.py painters/base_soms/lstm.pt      -hl -o painters/imgs/freqs/lstm.png      -d painters/dataset/ -m painters/model.pt -mc painters/model.py -n
python3 view_som.py painters/base_soms/hidden.pt    -hl -o painters/imgs/freqs/hidden.png    -d painters/dataset/ -m painters/model.pt -mc painters/model.py -n

