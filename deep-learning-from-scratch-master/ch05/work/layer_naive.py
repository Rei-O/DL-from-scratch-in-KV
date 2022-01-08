import numpy as np

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