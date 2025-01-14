import torch
import numpy as np
import os
import sys

count=1
ts = []
while os.path.isfile("agenet_y_"+str(count)):
        print("   ",count)
        sys.stdout.flush()
        bts = torch.load("agenet_y_"+str(count))        
        for v in bts:
            ts.append(int(v))
        count+=1

print(len(ts))
print(np.array(ts).mean())
print(np.array(ts).std())

from sklearn.preprocessing import KBinsDiscretizer
est = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy="kmeans")
res = est.fit_transform(np.array([ts]).T)
print(np.count_nonzero(res == 0))
print(np.count_nonzero(res == 1))
print(np.count_nonzero(res == 2))

torch.save(res, "agenet_y_d")
