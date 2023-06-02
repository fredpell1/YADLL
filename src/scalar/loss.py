from src.scalar.autodiff import Scalar
import numpy as np

def MSE(y_pred, y) -> Scalar:
    return 1/len(y_pred) * sum((y_p - y_gt) **2 for y_p, y_gt in zip(y_pred, y))