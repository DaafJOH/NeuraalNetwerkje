import numpy as np
import struct
import matplotlib.pyplot as plt

with open('DataSet.idx3-ubyte','rb') as f:
    magic, size = struct.unpack(">II", f.read(8))
    nrows, ncols = struct.unpack(">II", f.read(8))
    data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
    data = data.reshape((size, nrows, ncols))

with open("DataSetLabels.idx1-ubyte", "rb") as f:
    dataL = list(f.read())[8:]
print(dataL[4565])
plt.imshow(data[4565])
plt.show()
