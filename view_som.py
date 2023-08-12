import sys
import os
from actsom import ActSom

if len(sys.argv) < 2:
    print("Provide either:")
    print("   - a pickle file with a SOM saved, or")
    print("   - a directory where to fine a set of SOMs and one where the save the images")
    sys.exit(-1)
elif len(sys.argv)==2:
    asom = ActSom(sys.argv[1])
    print(asom.grid)
    print(asom.amap)
    asom.display()
elif len(sys.argv)==3:
    fs = os.listdir(sys.argv[1])
    for f in fs:
        infile = sys.argv[1]+f
        outfile = sys.argv[2]+f[:f.rindex(".")]+".png"
        print(f)
        asom = ActSom(infile)
        asom.display(outfile=outfile)
