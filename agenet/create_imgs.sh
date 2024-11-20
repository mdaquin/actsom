# /bin/bash

l=`cd agenet/base_soms/; ls *.pt; cd -`

for f in $l 
do
   python3 view_som.py agenet/base_soms/$f -hl -o agenet/imgs/soms/$f.png
done

python3 view_som.py agenet/base_soms/net.conv1.pt -hl -o agenet/imgs/freqs/net.conv1.png -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/net.layer1.pt -hl -o agenet/imgs/freqs/net.layer1.png -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/net.layer2.pt -hl -o agenet/imgs/freqs/net.layer2.png -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/net.layer3.1.bn2.pt -hl -o agenet/imgs/freqs/net.layer3.1.bn2.png -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/net.layer4.1.bn2.pt -hl -o agenet/imgs/freqs/net.layer4.1.bn2.png -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/net.pt        -hl -o agenet/imgs/freqs/net.png        -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/linear.pt     -hl -o agenet/imgs/freqs/linear.png     -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n
python3 view_som.py agenet/base_soms/out.pt     -hl -o agenet/imgs/freqs/out.png     -d agenet/dataset/ -m agenet/agenet_4.75_0.88 -mc agenet/model.py -n

