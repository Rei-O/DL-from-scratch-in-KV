import numpy as np

def softmax(x):
    # オーバーフロー対策
    x -= np.max(x, axis=1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def cross_entropy_error(x, t):
    # 1次元データの場合
    if x.ndim == 1:
        # 2次元配列に整形
        x = x.reshape(1, x.size)
        t = t.reshape(1, t.size)

    # 正解データがone-hot表現の場合
    if t.size == x.size:
        # ラベル表現に変換
        t = t.argmax(axis=1)
    
    # バッチサイズ取得
    batch_size = x.shape[0]

    # 交差エントロピー算出
    return np.sum(np.log(x[np.arange(batch_size).astype(int), t.astype(int)] + 1e-7)) / batch_size