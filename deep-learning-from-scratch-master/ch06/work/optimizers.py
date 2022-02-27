import numpy as np

class SGD:
    """
    ### SGD class
    #### ■概要
    確率勾配降下法によるパラメータ更新を行うクラス
    #### ■メソッド
    update : (params, grads) -> None
    """
    def __init__(self, lr=0.1):
        self.lr=lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]

class Momentum:
    """
    ### Momentum class
    #### ■概要
    モーメンタムによるパラメータ更新を行うクラス
    #### ■メソッド
    update : (params, grads) -> None
    """
    def __init__(self, lr=0.1, momentum=0.9):
        """
        lr:学習率
        momentum:速度減衰率（摩擦や空気抵抗に相当）
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """
    ### AdaGrad class
    #### ■概要
    AdaGradによるパラメータ更新を行うクラス
    #### ■メソッド
    update : (params, grads) -> None
    """
    def __init__(self, lr=0.1):
        self.lr = lr
        self.h = None
    
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7) #微小量はゼロ除算回避のため