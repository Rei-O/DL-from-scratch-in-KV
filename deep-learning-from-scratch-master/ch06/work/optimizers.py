class SGD:
    """
    ### SGD class
    #### ■概要
    確率勾配降下法によるパラメータ更新を行うクラス
    #### ■メソッド
    update : (self, params, grads) -> None
    """
    def __init__(self, lr=0.9):
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
    update : (self, params, grads) -> None
    """

class AdaGrad:
    """
    ### AdaGrad class
    #### ■概要
    AdaGradによるパラメータ更新を行うクラス
    #### ■メソッド
    update : (self, params, grads) -> None
    """