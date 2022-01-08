# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
import numpy as np

from common.sample.functions import softmax, cross_entropy_error
from common.sample.gradient import numerical_gradient


class simpleNet:
    """
    ### simpleNet class
    #### ■概要
    勾配法の実装をするための簡単なニューラルネットワーク
    #### ■メソッド
    predict : (self, x) -> Array
    loss : (self, x, t) -> float
    """
    def __init__(self):
        self.W = np.random.randn(2,3) # 重みの2*3行列をガウス分布からランダム生成
    
    # 予測
    def predict(self, x):
        return np.dot(x, self.W)
    
    # 損失関数
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y,t)

        return loss


if __name__ == '__main__':
    # simpleNetのインスタンス生成
    net = simpleNet()
    # 重みパラメータの初期値を確認
    print(net.W)

    # 予測データ
    x = np.array([0.6, 0.9])
    # 予測を実行
    p = net.predict(x)
    # 予測結果を確認
    print(p)
    # 分類問題の場合の予測ラベルを確認
    print(np.argmax(p))

    # 正解ラベル
    t = np.array([0, 0, 1])

    # 勾配降下法
    # numerical_gradientに渡すための関数オブジェクトを定義
    f = lambda w: net.loss(x, t)

    # 勾配降下法の実施
    dW = numerical_gradient(f, net.W)
    print(dW)