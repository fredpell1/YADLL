#set the backend to numpy
import os
os.environ['backend'] = 'numpy'

from yadll.ir import *
import numpy as np


def test_add():
    a = Data([1,2,3])
    b = Data([1,2,3])
    assert add == np.add, "operation not the same"
    assert np.all(add(a,b) == (a+b)), "result not the same"