import numpy as np

class Adam:
    def __init__(self, lr=0.0001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1  # 1次モーメントの減衰率
        self.beta2 = beta2  # 2次モーメントの減衰率
        self.iter = 0
        self.m = None  # 1次モーメント
        self.v = None  # 2次モーメント

    def update(self, params, grads):
        # 1次モーメントがNone(初期化後)の場合
        if self.m is None:
            # 初期値設定
            self.m, self.v = {}, {}            
            for key, val in params.items():
                    self.m[key] = np.zeros_like(val)
                    self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            self.m[key] += (1.0 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1.0 - self.beta2) * (grads[key] ** 2 - self.v[key]) 

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

optimizer_class_dict={'adam':Adam}
