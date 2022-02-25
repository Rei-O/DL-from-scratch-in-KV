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
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        """
        Abstract
        ---------------
        BatchNormレイヤのforward, backwardを行う. 

        Params
        ---------------
        """
        # forward
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv層は4次元、全結合層は2次元（になるらしい）

        # 予測時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, X, train_flg=True):
        self.input_size = X.shape
        # 4次元の場合（Conv層は4次元、全結合層は2次元を想定）
        if X.ndim != 2:
            # N = X.shape[0]  # これでも良い気がする
            N, C, H, W = X.shape
            X = X.shape(N, -1)  # 2次元に成形
        
        # 初回学習時の場合
        if self.running_mean is None:
            # D = X.shape[1]  # これでも良い気がする
            N, D = X.shape
            # 初期値設定
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)
        
        # 学習時
        if train_flg:
            out = self.__train(X)
        # 予測時
        else:
            out = self.__predict(X)

        return out

    def __train(self, X):
        mu = X.mean(axis=0)  # 平均
        xc = X - mu
        var = np.mean(xc**2, axis=0)  # 分散
        std = np.sqrt(var + 10e-7)  # 標準偏差
        xn = xc / std  # 正規化

            # backwardに使用する値を格納
        self.batch_size = X.shape[0]
        self.xc = xc
        self.std = std
        
        # running_mean, running_varを更新
        self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
        self.running_var = self.momentum * self.running_var + (1-self.momentum) * var

        out = self.gamma * xn + self.beta
        return out

    def __predict(self, X):
        mu = self.running_mean
        xc = X - mu
        var = self.running_var
        std = np.sqrt(var + 10e-7)
        xn = xc / std

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        # 4次元の場合（Conv層は4次元、全結合層は2次元を想定）
        if dout.ndim != 2:
            # N = dout.shape[0]
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dbeta = dout.sum(axis=0)
        dgammna = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dstd = -np.sum((self.xc * self.dxn) / (self.std ** 2), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar  # ここだけ分からず
        dmu = -np.sum(dxc, axis=0)
        dx = dxc + dmu / self.batch_size

        self.dgamma = dgammna
        self.dbeta = dbeta

        return dx


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
        self.y = None
        self.t = None
        self.loss = None

        # backward

    def forward(self, X, t):
        self.y = softmax(X)
        self.t = t
        self.loss =  crossEntropyError(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 教師データがone-hot表現の場合
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        # 教師データがラベル表現の場合
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1  # 正解ラベルの値は1が立っていると解釈して予測値から1を引く
            dx = dx / batch_size

        return dx