import numpy as np
from numpy.core.shape_base import _atleast_1d_dispatcher, atleast_1d

#================================
# 1層ずつ実装する
#================================


#====活性化関数：シグモイド関数==== 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#==========第1層==========
# 入力データ
X = np.array([0.5, 0.2])

# 重み
W1 = np.array([[0.2, 0.1, 0.5],[0.4, 0.2, 0.6]])

# バイアス
B1 = np.array([0.2, 0.4, 0])


# 出力結果
A1 = np.dot(X, W1) + B1
print(f"A1 : {A1}")

Z1 = sigmoid(A1)
print(f"Z1 : {Z1}")



#==========第2層===========
# 重み
W2 = np.array([[0.3, 0.2], [0.2, 0.7], [1.0, 0.6]])

# バイアス
B2 = np.array([1.0, 0.2])


# 出力結果
A2 = np.dot(Z1, W2) + B2
print(f"A2 : {A2}")

Z2 = sigmoid(A2)
print(f"Z2 : {Z2}")



#==========出力層==========
# 重み
W3 = np.array([[0.5, 0], [0.1, 0.7]])

# バイアス
B3 = np.array([1.0, 0.2])


# 出力結果
A3 = np.dot(Z2, W3) + B3
print(f"A3 : {A3}")

y = sigmoid(A3)
print(f"y : {y}")



#================================
# 出力層の設計
#================================

# ソフトマックス関数
def softmax(X):
    c = np.max(X)
    exp_X = np.exp(X-c)
    sum_exp_X = np.sum(exp_X)
    return exp_X / sum_exp_X



#================================
# 上記をまとめて関数化する
#================================

# ニューラルネットワークの初期化
def init_nuralnetwork():
    # 各層のパラメータを格納するdict
    network = {}

    #==========第1層==========
    # 重み
    network["W1"] = np.array([[0.2, 0.1, 0.5],[0.4, 0.2,0.6]])
    # バイアス
    network["B1"] = np.array([0.2, 0.4, 0])

    #==========第2層===========
    # 重み
    network["W2"] = np.array([[0.3, 0.2], [0.2, 0.7], [1.0, 0.6]])
    # バイアス
    network["B2"] = np.array([1.0, 0.2])

    #==========出力層==========
    # 重み
    network["W3"] = np.array([[0.5, 0], [0.1, 0.7]])
    # バイアス
    network["B3"] = np.array([1.0, 0.2])

    return network

# フォワード関数
def forward(network, X):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    B1, B2, B3 = network["B1"], network["B2"], network["B3"]

    def _forward(X, W, B):
        return sigmoid(np.dot(X, W) + B)

    # 第1層
    Z1 = _forward(X, W1, B1)

    # 第2層
    Z2 = _forward(Z1, W2, B2)

    # 出力結果
    Z3 = _forward(Z2, W3, B3)
    y = softmax(Z3)

    return y

# 入力データ
X = np.array([0.5, 0.2])

# ニューラルネットワーク初期化
network = init_nuralnetwork()

# フォワード
y = forward(network, X)

print(y)


