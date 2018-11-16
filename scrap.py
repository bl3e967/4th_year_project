import numpy as np

a = np.array([[1,2],[3,4]])
b = np.stack([a]*4,axis=2)
print(b)