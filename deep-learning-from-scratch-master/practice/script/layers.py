import numpy as np
import abc
from functions import softmax, crossEntropyError

############################## Sample ##############################

class Sample:
    def __init__(self):
        """
        Abstract
        ---------------
        レイヤのforward, backwardを行う. 

        Params
        ---------------
        """
        # forward

        # backward

    def forward(self, X, train_flg=True):
        pass

    def backward(self, dout):
        pass

####################################################################

########################### Abstract Class ###########################

class Layer(metaclass=abc.ABCMeta):
    """
    通常レイヤの抽象クラス
    """
    @abc.abstractmethod
    def forward(self, X, train_flg=True):
        pass

    @abc.abstractmethod
    def backward(self, dout):
        pass


class LossLayer(metaclass=abc.ABCMeta):
    """
    損失の計算を行うレイヤの抽象クラス
    """
    @abc.abstractmethod
    def forward(self, X, t):
        pass

    @abc.abstractmethod
    def backward(self, dout=1):
        pass
  


########################### Layer Class ###########################


class Affine(Layer):
    def __init__(self, W, b):
        """
        Abstract
        ---------------
        Affineレイヤのforward, backwardを行う. 

        Params
        ---------------
        W : 重み
        b : バイアス
        """
        # forward
        self.W = W
        self.b = b
        self.X = None
        self.org_X_shape = None  # テンソル対応

        # backward
        self.dW = None
        self.db = None

    def forward(self, X, train_flg=True):
        self.org_X_shape = X.shape
        self.X = X.reshape(X.shape[0], -1)  # テンソル対応
        self.X = X
        Z = np.dot(self.X, self.W) + self.b
        return Z

    def backward(self, dout):
        self.dW = np.dot(self.X.T,dout)
        self.db = dout
        dX = np.dot(dout, self.W.T)
        dX = dX.reshape(*self.org_X_shape)  # テンソル対応で形状変更したため元に戻す
        return dX


class ReLU(Layer):
    def __init__(self):
        """
         Abstract
        ---------------
        ReLUレイヤのforward, backwardを行う. 


        Params
        ---------------
        None
        """
        # forward
        self.mask = None

        # backward

    def forward(self, X, train_flg=True):
        self.mask = (X > 0)
        Z = self.mask * X
        return Z

    def backward(self, dout):
        dX = self.mask * dout
        return dX


class Softmax(Layer):
    def __init__(self):
        """
        Abstract
        ---------------
        Softmaxレイヤのforward, backwardを行う. 

        Params
        ---------------
        """
        # forward
        self.Y = None
        self.Z = None

        # backward

    def forward(self, X, train_flg=True):
        self.Z = softmax(X)
        return self.Z

    def backward(self, dout):
        return dout * self.Z * (1 - self.Z)

class BatchNorm(Layer):
    def __init__(self):
        """
        Abstract
        ---------------
        BatchNormレイヤのforward, backwardを行う. 

        Params
        ---------------
        """
        # forward

        # backward

    def forward(self, X, train_flg=True):
        pass

    def backward(self, dout):
        pass

class Dropout(Layer):
    def __init__(self, dropout_ratio = 0.5):
        """
        Abstract
        ---------------
        Dropoutレイヤのforward, backwardを行う. 

        Params
        ---------------
        dropout_ratio : ドロップアウト率
        """
        # forward
        self.dropout_ratio = dropout_ratio
        self.mask = None

        # backward

    def forward(self, X, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*X.shape) > self.dropout_ratio
            return X * self.mask
        else:
            return X * (1.0 - self.dropout_ratio)  # 予測時は全ノードから出力。出力スケールを学習時と合わせるため重みづけ          

    def backward(self, dout):
        return dout * self.mask

class CrossEntropy(LossLayer):
    def __init__(self):
        """
        Abstract
        ---------------
        CrossEntropyレイヤのforward, backwardを行う. 

        Params
        ---------------
        """
        # forward
        self.X = None
        self.t = None

        # backward

    def forward(self, X, t):
        self.X = X
        self.t = t
        return crossEntropyError(self.X, self.t)

    def backward(self, dout=1):
        return dout * (- self.t / self.X)

class SoftmaxWithLoss(LossLayer):
    def __init__(self):
        """
        Abstract
        ---------------
        SoftmaxWithLossレイヤのforward, backwardを行う. 

        Params
        ---------------
        """
        # forward
        self.X = None
        self.t = None
        self.Z = None

        # backward

    def forward(self, X, t):
        self.X = X
        self.t = t

        return crossEntropyError(self.X, self.t)

    def backward(self, dout=1):
        return dout * (- self.t / self.X)