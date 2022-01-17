import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
import common.presentation.debug as debug
from common.sample.functions import softmax, cross_entropy_error

class MulLayer:
    """
    ### MulLayer class
    #### ■概要
    乗算レイヤの順伝播、逆伝播を行うためのクラス
    #### ■メソッド
    forward : (x, y) -> Array
    backward : (dout) -> Array, Array
    """
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out
    
    def backward(self, dout):
        # 上流の微分にxとyを逆に掛けて下流に渡す
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy
    
class AddLayer:
    """
    ### AddLayer class
    #### ■概要
    加算レイヤの順伝播、逆伝播を行うためのクラス
    #### ■メソッド
    forward : (x, y) -> Array
    backward : (dout) -> Array, Array
    """
    def __init__(self):
        self.x = None
        self.y = None
    
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x + y
        return out
    
    def backward(self, dout):
        # 上流の微分をそのまま下流に渡す
        dx = dout * 1
        dy = dout * 1
        return dx, dy
    

class Relu:
    """
    ### Relu class
    #### ■概要
    ReLu関数の順伝播、逆伝播を行うためのクラス
    #### ■メソッド
    forward : (x) -> Array
    backward : (dout) -> Array, Array
    """
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x <= 0) # maskに0以下をTrue,1より大きい値をFalseでもつ配列を格納
        out = x.copy() 
        out[self.mask] = 0 # 0以下の要素を0に更新する
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0 # 0以下の要素を0に更新する
        dx = dout * 1
        return dx 


class Sigmoid:
    """
    ### Sigmoid class
    #### ■概要
    Sigmoid関数の順伝播、逆伝播を行うためのクラス
    #### ■メソッド
    forward : (x) -> Array
    backward : (dout) -> Array, Array
    """
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out
    
    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx 

class Affine:
    """
    ### Affine class
    #### ■概要
    アフィンレイヤの順伝播、逆伝播を行うためのクラス
    #### ■メソッド
    forward : (x) -> Array
    backward : (dout) -> Array, Array
    """
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T) #.Tは転置行列
        self.dW = np.dot(self.x.T, dout) #.Tは転置行列
        self.db = np.sum(dout, axis=0)
        return dx


class SoftmaxWithLoss:
    """
    ### SoftmaxWithLoss class
    #### ■概要
    分類問題における出力レイヤ（ソフトマックス関数による正規化⇒誤差算出）の順伝播、逆伝播を行うためのクラス
    #### ■メソッド
    forward : (x) -> Array
    backward : (dout) -> Array, Array
    """
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    
    def forward(self, x, t):
        self.x = x
        self.y = softmax(x)
        self.t = t
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):   
        batch_size = self.t.shape 
        dx = (self.y -self.t) / batch_size
        return dx

if __name__ == '__main__':
    debug.isDebugMode = False

    W = np.array([[1,2,3], [4,5,6]])
    b = np.array([0.1, 0.1, 0.3])
    X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    net = Affine(W, b)
    Y = net.forward(X)
    debug.debugprt(Y, "Y")

    dY = np.array([[0.1, 1, -1], [0.2, 0.1, 1], [1, 0.3, -1], [0.4, 2, 1]])
    dx = net.backward(dY)
    debug.debugprt(dx ,"dx")
    debug.debugprt(net.dW, "net.dW")
    debug.debugprt(net.db, "net.db")
