import numpy as np
import sys, os
# from nural_network import sigmoid, softmax
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

# 学習データと予測データを取得
(x_train, t_train) ,(x_test, t_test) = load_mnist(flatten=True, normalize=False)


# 学習データサイズを取得
train_size = x_train.shape[0]

# バッチサイズを指定
batch_size = 100

# ミニバッチ用データを取得
batch_mask = np.random.choice(train_size, batch_size) # train_sizeまでの数字の中からbatch_size分だけランダムに数字を取得
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


# 交差エントロピー誤差関数 (tはone-hot表現)
def cross_entropy_error(y, t):
    delta = 1e-7 # log(0)=-infとなることを防ぐための微小値

    # 出力値(y)がベクトル(1つの学習データ)か判定する変数
    is_vector = False

    # 配列の場合縦横変換(n×1⇒1×n行列)
    if y.ndim == 1:
        y.reshape(1, y.size)
        t.reshape(1, t.size)
        is_vector = True
    
    # 正解データがone-hot表現か判定する変数
    is_one_hot = False

    # 出力値(y)が2次元以上(複数の学習データ)かつ正解データが2次元の場合
    if is_vector == False and t.ndim == 2:
        is_one_hot = True
    # 出力値(y)がベクトル(1つの学習データ)かつ正解データがの長さが2以上の場合
    elif is_vector == True and t.shape[0] > 1:
        is_one_hot = True

    batch_size = y.shape[0]

    # targetの配列がone-hot表現の場合
    if is_one_hot:
        # print("bebug : one-hot")
        return -np.sum(t * np.log(y+delta)) / batch_size
    # targetの配列が正解ラベルの場合
    elif is_vector:
        # print("debug : label & vector")
        # 正解ラベルに対応するモデルの出力値だけを抽出して損失を計算
        return -np.sum(np.log(y[t]+delta)) / batch_size 
    else:
        # print("debug : label & matrix")
        # 正解ラベルに対応するモデルの出力値だけを抽出して損失を計算
        return -np.sum(np.log(y[np.arange(batch_size), t]+delta)) / batch_size 

print(cross_entropy_error(x_batch, t_batch))

# データ1つ・正解データがone-hot表現の場合
Y1_1 = np.array([0.2,0.1,0.7])
T1_1 = np.array([1,0,0])

Y1_2 = np.array([0.5,0.2,0.3])
T1_2 = np.array([1,0,0])

Y1_3 = np.array([0.2,0.4,0.4])
T1_3 = np.array([0,0,1])
print(cross_entropy_error(Y1_1, T1_1)+cross_entropy_error(Y1_2, T1_2)+cross_entropy_error(Y1_3, T1_3))

# データ複数・正解データがone-hot表現の場合
Y2 = np.array([[0.2,0.1,0.7],[0.5,0.2,0.3],[0.2,0.4,0.4]])
T2 = np.array([[1,0,0],[1,0,0],[0,0,1]])
print(cross_entropy_error(Y2, T2))

# データ複数・正解データがラベル表現の場合
Y3 = np.array([[0.2,0.1,0.7],[0.5,0.2,0.3],[0.2,0.4,0.4]])
T3 = np.array([0,0,2])
print(cross_entropy_error(Y3, T3))

# データ1つ・正解データがラベル表現の場合
Y4_1 = np.array([0.2,0.1,0.7])
T4_1 = np.array([0])

Y4_2 = np.array([0.5,0.2,0.3])
T4_2 = np.array([0])

Y4_3 = np.array([0.2,0.4,0.4])
T4_3 = np.array([2])
print(cross_entropy_error(Y4_1, T4_1)+cross_entropy_error(Y4_2, T4_2)+cross_entropy_error(Y4_3, T4_3))