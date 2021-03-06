import sys
sys.path.append('python-packages')

from types import SimpleNamespace

import numpy as np
import nrrd

data, header = nrrd.read('utils/cube.nrrd')
print(data.shape)
print(header)

mat = np.append(header['space directions'], header['space origin'][:, np.newaxis], axis=1)
ItoW = np.append(mat, [[0, 0, 0, 1]], axis=0)
volume = SimpleNamespace(data=data, ItoW=ItoW)
print(volume.data.shape)
print(volume.data[:10])

data, header = nrrd.read('utils/lut.nrrd')
print(data.shape)
print(header)

txf_min = header['axis mins'][1]
txf_max = header['axis maxs'][1]
txf = SimpleNamespace(data=data, txf_min=txf_min, txf_max=txf_max)
