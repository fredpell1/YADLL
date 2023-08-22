from typing import Union
from numbers import Number
class Data: pass

def add(data1: Data, data2: Data, *args, **kwargs) -> Data: None
def mul(data1: Data, data2: Union[Data,Number], *args, **kwargs) -> Data: None
def matmul(data1: Data, data2: Data, *args, **kwargs) -> Data: None
def pow(data: Data, power: Union[Data, Number], *args, **kwargs) -> Data: None
