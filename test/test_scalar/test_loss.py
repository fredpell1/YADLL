from src.scalar.loss import *

def test_MSE():
    y_pred = [1,1]
    y = [2,2]
    out = MSE(y_pred, y)
    expected_out = 0.5 * (1 + 1)
    assert out == expected_out