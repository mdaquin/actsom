# /bin/bash

python3 view_som.py agenet/base_soms/net.layer1.1.bn2.pt -hl -o agenet/imgs/soms/net.layer1.png
python3 view_som.py agenet/base_soms/net.layer2.1.bn2.pt -hl -o agenet/imgs/soms/net.layer2.png
python3 view_som.py agenet/base_soms/net.layer3.1.bn2.pt -hl -o agenet/imgs/soms/net.layer3.png
python3 view_som.py agenet/base_soms/net.layer4.1.bn2.pt -hl -o agenet/imgs/soms/net.layer4.png
python3 view_som.py agenet/base_soms/net.pt        -hl -o agenet/imgs/soms/net.png
python3 view_som.py agenet/base_soms/linear.pt     -hl -o agenet/imgs/soms/linear.png
python3 view_som.py agenet/base_soms/relu.pt       -hl -o agenet/imgs/soms/relu.png

python3 view_som.py agenet/base_soms/net.layer1.1.bn2.pt -hl -o agenet/imgs/freqs/net.layer1.png -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/net.layer2.1.bn2.pt -hl -o agenet/imgs/freqs/net.layer2.png -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/net.layer3.1.bn2.pt -hl -o agenet/imgs/freqs/net.layer3.png -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/net.layer4.1.bn2.pt -hl -o agenet/imgs/freqs/net.layer4.png -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/net.pt        -hl -o agenet/imgs/freqs/net.png        -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/linear.pt     -hl -o agenet/imgs/freqs/linear.png     -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/relu.pt       -hl -o agenet/imgs/freqs/relu.png       -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n

