import numpy as np
import glob
import sys

pathN = sys.argv[1]

files = glob.glob(pathN + "/lda_doc*.npy")

for name in files:
    a = np.load(name)
    print(name, '\n', a)

print('Done!')
