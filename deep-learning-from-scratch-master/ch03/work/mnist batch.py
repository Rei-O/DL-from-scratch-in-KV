import numpy as np
import sys, os
from nural_network import sigmoid, softmax
import pickle

# リスト（list型）や辞書（dict型）などのオブジェクトをきれいに整形して出力・表示するパッケージ
import pprint

# 親ディレクトリのファイルをインポートするための設定
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 親ディレクトリの親ディレクトリのファイルをインポートするための設定
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))

# 探索パスを出力
# pprint.pprint(sys.path)

from dataset.presentation.mnist import load_mnist

# 予測データと正解ラベルを取得する
def get_data():
    (x_train, t_train) ,(x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test

# ニューラルネットワークの重みとバイアスをsample_weight.pklから読み込み初期化
def init_network():
    with open(os.path.join(os.path.dirname(__file__), 'sample_weight.pkl'), "rb") as f:
        network = pickle.load(f)
    return network

# 予測を行う関数
def predict(network, X):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    def _forward(X, W, B):
        return sigmoid(np.dot(X, W) + B)

    # 第1層
    Z1 = _forward(X, W1, b1)

    # 第2層
    Z2 = _forward(Z1, W2, b2)

    # 出力結果
    Z3 = _forward(Z2, W3, b3)
    y = softmax(Z3)

    return y


# バッチ処理により同時に複数処理を行う。
# バッチ処理は以下の理由により高速化になる
# 　・nupmyには大きな配列の計算最適化があり
# 　・データ読込のオーバーヘッドも短縮できる
if __name__ == '__main__':
    x, t = get_data()
    net_work = init_network()

    # バッチサイズ
    batch_size = 100
    # 正解数
    accuracy_cnt = 0

    for i in range(0, len(x), batch_size): # range(初期値, 最大値, ステップ数)
        x_batch = x[i:i+batch_size]
        # 予測
        y_batch = predict(net_work, x_batch)
        # 最も確率が高い要素のインデックスを取得
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])

    print("Accuracy : " + str(float(accuracy_cnt/len(x))))
