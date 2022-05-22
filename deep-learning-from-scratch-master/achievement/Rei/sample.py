
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

# 0~9の手書き文字MNISTのデータセットを読み込む
(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# 画像データの形式を変更する
training_images = training_images.reshape(training_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
# 画像データを正規化する
training_images = training_images / 255.0
test_images = test_images / 255.0
# ラベルデータを1-of-K表現にする
training_labels = tf.keras.utils.to_categorical(training_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)




# CNNのモデルを作成する
model = tf.keras.models.Sequential()
model.add(Input(shape=(28, 28, 1)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

# 任意のオプティマイザと損失関数を設定してモデルをコンパイルする
model.compile(optimizer='adam', loss='categorical_crossentropy' \
    , metrics=['accuracy'])

# モデルをトレーニングする
model.fit(training_images, training_labels, epochs=5)

# テストデータで精度を確認する
test_loss = model.evaluate(test_images, test_labels)
