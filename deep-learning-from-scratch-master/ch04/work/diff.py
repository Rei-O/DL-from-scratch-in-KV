import numpy as np
import matplotlib.pylab as plt

# 前方微分
def numerical_diff_forward(f, x):
    h = 1e-4
    return (f(x + h) - f(x)) / h

# 後方微分
def numerical_diff_backward(f, x):
    h = 1e-4
    return (f(x) - f(x-h)) / h

# 中央微分
def numerical_diff_center(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

# 接線
def tangent_line(diff, f, x):
    d = diff(f, x)
    # print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

def quadratic_func(x):
    return x**2-3*x+4

print(f"前方微分 : {numerical_diff_forward(quadratic_func, 12)}")
print(f"後方微分 : {numerical_diff_backward(quadratic_func, 12)}")
print(f"中央微分 : {numerical_diff_center(quadratic_func, 12)}")

x = np.arange(10, 15, 0.1)
y = quadratic_func(x)

tf_forward = tangent_line(numerical_diff_forward, quadratic_func, 12)
y2 = tf_forward(x)

tf_backward = tangent_line(numerical_diff_backward, quadratic_func, 12)
y3 = tf_backward(x)

tf_center = tangent_line(numerical_diff_center, quadratic_func, 12)
y4 = tf_center(x)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.plot(x,y2)
plt.plot(x,y3)
plt.plot(x,y4)
plt.show()