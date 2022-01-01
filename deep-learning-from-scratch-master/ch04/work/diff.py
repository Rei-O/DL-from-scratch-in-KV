import numpy as np
import matplotlib.pylab as plt

# 前方微分
def numerical_diff_forward(f, x):
    h = 10e-4
    return (f(x + h) - f(x)) / h

# 後方微分
def numerical_diff_backward(f, x):
    h = 10e-4
    return (f(x) - f(x-h)) / h

# 中央微分
def numerical_diff_center(f, x):
    h = 10e-4
    return (f(x+h) - f(x-h)) / 2*h

def quadratic_func(x):
    return 2*x**2-3*x+4

print(f"前方微分 : {numerical_diff_forward(quadratic_func, 2)}")
print(f"後方微分 : {numerical_diff_backward(quadratic_func, 2)}")
print(f"中央微分 : {numerical_diff_center(quadratic_func, 2)}")

