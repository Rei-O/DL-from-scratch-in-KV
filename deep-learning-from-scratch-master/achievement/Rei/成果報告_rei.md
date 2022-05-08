---
marp: true
theme: gaia
header: "ゼロつく成果報告 @れい"
footer: "機械学習勉強会 in KinoVillage"
paginate: true
---
# ゼロつく成果報告 @れい
<!--
_class: lead
_paginate: false
_header: ""
_footer: ""
-->
---
## 目標

* 本で学んだことを自身に定着させる。
* 分かったこと、分からないことを境界線を知る。

## 取り組むこと

* 自分用のNNWモジュールを作成する
　→kerasライクな実装に書き換えてみる
* 自分用モジュールでどこかコンペに投稿

---
## ルール

* サンプルのコピペNG（ただし自身が手打ちしたものをコピペするのはOK）
　→何となく分かった気で実装しないように
* 自分が理解していないことを実装するのはNG
　→分かっていないことは素直に認める

---
## Kerasの使い方


```python
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

# データ加工処理は省略

# モデルの作成
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
 optimizer=RMSprop(),
 metrics=['accuracy'])

# 学習は、scrkit-learnと同様fitで記述できる
history = model.fit(x_train, y_train,
 batch_size=batch_size,
 epochs=epochs,
 verbose=1,
 validation_data=(x_test, y_test))

# 評価はevaluateで行う
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```


---

<br>
<br>
<br>
<br>
<br>
ご清聴ありがとうございました。