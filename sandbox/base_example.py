from yadll.tensor.autodiff import *

a = Tensor.random((10, 20, 20))
b = Tensor.random((10, 20, 20))
c = (a @ b).sum()
print(c)
