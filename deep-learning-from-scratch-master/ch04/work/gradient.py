import numpy as np
import decimal
import fractions

# 数値勾配法
def numerical_gradient(f,x):
    """
    param : Object, Array
    return : Array 
    勾配を求める関数と座標を渡し、勾配を配列形式で返却する
    """
    # 微小量
    h = 1e-4

    # 勾配を格納する配列
    grad = np.zeros_like(x) # xと同じようなゼロ配列を生成

    # 変数ごとに偏微分
    for idx in range(x.size):
        # 変数の値を一時的に格納する変数
        tmp_val = x[idx]

        # f(x+h)の計算
        x[idx] = tmp_val + h # x[idx]のみ微小量を加算
        fxh1 = f(x)

        # f(x-h)の計算
        x[idx] = tmp_val - h # x[idx]のみ微小量を減算
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 値を元に戻す
    
    return grad

# 勾配降下法
def gradient_descent(f, init_x, learning_rate=0.01, step_num=100):
    """
    param : f=Object, init_x=Array, learning_rate=float(init=0.01), step_num=int(init=100) 
    return : Array
    """
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x = x - learning_rate * grad
    
    return x


def func_1(x):
    return x[0]**2+x[0]*x[1]-3*x[1]**2+4

def func_2(x):
    return x[0]**2+x[1]**2


print(f"数値勾配 : {numerical_gradient(func_2, np.array([3.0,4.0]))}")

print(f"勾配降下法(learning_rate=0.1) : {gradient_descent(func_2, np.array([3.0,4.0]), 0.1)}")

# 学習率が大きすぎて発散する例
print(f"勾配降下法(learning_rate=10.0) : {gradient_descent(func_2, np.array([3.0,4.0]), 10.0)}")

# 学習率が小さすぎて学習が進まない例
print(f"勾配降下法(learning_rate=1.0^-10) : {gradient_descent(func_2, np.array([3.0,4.0]), 1e-10)}")