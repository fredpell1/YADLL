from src.scalar.optimizer import *
from src.scalar.nn import *
def test_SGD():
    x = [1.2,-1]
    model = Sequential(
        Layer(2,2),
        Layer(2,1,activation_func = 'sigmoid')
    )
    optim = SGD(model.parameters(), 0.1)
    out = model(x)[0]
    w0_before = model.parameters()[0].data
    out.backward()
    w0_grad = model.parameters()[0].grad
    optim.step()
    w0_after = model.parameters()[0].data
    assert abs(w0_after - (w0_before - 0.1*w0_grad)) < 0.0000001
