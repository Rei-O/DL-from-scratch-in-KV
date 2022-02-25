import numpy as np


def relu(x):
    """
    ReLU関数

    Params
    ---------------
    x : 入力値
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Softmax関数

    Params
    ---------------
    x : 入力値
    """
    # オーバーフロー対策
    x -= np.max(x, axis=-1, keepdims=True)  # 入力値のうち最大値を減する

    # ソフトマックス算出
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def crossEntropyError(y, t):
    """
    交差エントロピー関数

    Params
    ---------------
    y : 予測値
    t : 教師データ
    """
    # 次元1の場合
    if y.ndim == 1:
        # 2次元配列に成形
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    # 教師データがone-hot表現の場合
    if t.size == y.size:  # 要素数が等しい場合（one-hot表現の場合）
        # ラベル表現に変換
        t = t.argmax(axis=1)

    # バッチサイズ
    batchSize = y.shape[0]

    # 交差エントロピー算出
    return -np.sum(np.log(y[np.arange(batchSize).astype(int), t.astype(int)] + 1e-7)) / batchSize