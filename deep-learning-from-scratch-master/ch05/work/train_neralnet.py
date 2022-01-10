# coding: utf-8
import sys, os
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.presentation.mnist import load_mnist
from two_layer_nuralnet import TwoLayerNet
import matplotlib.pylab as plt

# 画像データ読込み
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# ニューラルネットワークを生成
net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10) # 入力は28*28ピクセルの画像データ、手書き数字0~9の識別を想定


# ハイパーパラメータの設定
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1


# 学習過程を格納する配列
# 損失
train_loss_list = []
# 学習正解率
train_acc_list = []
# テスト正解率
test_acc_list = []


# 1エポックあたりの繰り返し数
# エポックとは単位。「訓練データをすべて使い切った時の回数」を1エポックと表現する。
iter_per_epoch = max(train_size/batch_size, 1)     #  1エポックの回数 = データ数/ミニバッチサイズ


for i in range(iters_num):
    # ミニバッチを取得
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 数値微分による勾配導出
    # grad = net.numerical_gradient(x_batch, t_batch)

    # 誤差逆伝播法による勾配導出
    grad = net.gradient(x_batch, t_batch)

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