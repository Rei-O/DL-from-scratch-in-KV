# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
import numpy as np

from common.sample.functions import *
from common.sample.gradient import numerical_gradient
from dataset.sample.mnist import load_mnist
import matplotlib.pylab as plt

####################################################################
# 4章の総まとめとして確率的勾配降下法による「学習」を実装する
# 
# <<方針>>
# 1. ミニバッチ
# 2. 勾配の算出
# 3. パラメータの更新
# 4. 1~3の繰り返し
####################################################################


# 2層ニューラルネットワーク
class TwoLayerNet:
    """
    ### TwoLayerNet class
    #### ■概要
    確率的勾配降下法による「学習」を実装するための2層ニューラルネットワーク
    #### ■メソッド
    predict : (self, x) -> Array
    loss : (self, x, t) -> float
    """
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        self.params = {}

        # 1層のパラメータを初期化
        # 重みの初期化
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size) # 各要素がN(0, 0.01)に従う(input_size,hidden_size)行列を生成
        # バイアスの初期化
        self.params['b1'] = np.zeros(hidden_size)

        # 2層のパラメータを初期化
        # 重みの初期化
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) # 各要素がN(0, 0.01)に従う(hidden_size, output_size)行列を生成
        # バイアスの初期化
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        """
        入力データに対してニューラルネットワークの予測結果を返却する
        param : x=Array
        return : Array
        """
        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        # 1層目
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)

        # 2層目
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    # x:入力データ, t:教師データ
    def loss(self, x, t):
        """
        入力データ、正解ラベルに対してモデルによる予測を行い、クロスエントロピーによる損失結果を返却する
        param : x=Array(入力データ), t=Array(教師データ)
        return : Array
        """
        return cross_entropy_error(self.predict(x), t)
    
    def accuracy(self, x, t):
        """
        入力データ、正解ラベルに対してモデルによる予測を行い、正解率(Accuracy)を返却する
        param : x=Array(入力データ), t=Array(教師データ)
        return : Array
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1) # axis=1は縦方向で最大値のインデックスを返却
        t = np.argmax(t, axis=1)

        # Accuracyの計算
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        """
        入力データ、正解ラベルに対してモデルによる予測を行い、重みとバイアスの勾配を計算する
        param : x=Array(入力データ), t=Array(教師データ)
        return : Array
        """        
        loss_W = lambda W : self.loss(x,t)

        # 重み、バイアスの勾配を格納するディクショナリ
        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1']) # 各要素がN(0, 0.01)に従う(input_size,hidden_size)行列を生成
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2']) # 各要素がN(0, 0.01)に従う(hidden_size, output_size)行列を生成
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

if __name__ == '__main__':

    # 画像データ読込み
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # 学習過程を格納する配列
    # 損失
    train_loss_list = []
    # 学習正解率
    train_acc_list = []
    # テスト正解率
    test_acc_list = []

    # ハイパーパラメータの設定
    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    # 1エポックあたりの繰り返し数
    # エポックとは単位。「訓練データをすべて使い切った時の回数」を1エポックと表現する。
    iter_per_epoch = max(train_size/batch_size, 1)     #  1エポックの回数 = データ数/ミニバッチサイズ

    net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10) # 入力は28*28ピクセルの画像データ、手書き数字0~9の識別を想定

    # x = np.random.rand(100, 784) # ダミーの入力データ
    # t = np.random.rand(100, 10) # ダミーの正解ラベル

    # grads = net.numerical_gradient(x,t)

    for i in range(iters_num):
        # ミニバッチを取得
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = net.numerical_gradient(x_batch, t_batch)

        # パラメータの更新
        for key in ('W1', 'b1', 'W2', 'b2'):
            net.params[key] -= learning_rate * grad[key]
        
        # 学習過程を記録
        loss = net.loss(x_batch, t_batch)
        train_loss_list.append(loss)
    
        # 1エポックごとに認識精度を計算
        if i % iter_per_epoch == 0:
            train_acc = net.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            test_acc = net.accuracy(x_test, t_test)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    plt.xlabel("回数")
    plt.ylabel("loss")
    plt.plot(range(iters_num),train_loss_list)
    plt.show()

    # グラフの描画
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(train_acc_list))
    plt.plot(x, train_acc_list, label='train acc')
    plt.plot(x, test_acc_list, label='test acc', linestyle='--')
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.show()