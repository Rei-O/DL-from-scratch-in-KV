from reras.models import NuralNet
from reras.layers import *
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
from dataset.presentation.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=True)

# TODO 現状出力層とその前層のアウトプットunitsが合ってないとエラーになる（denseとかしてないのでcompileのoutput_unitsが意味ない）











# CNNのモデルを作成する
model = NuralNet()
model.add(Input(x_train.shape[1:]))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(filter_size=(2,2),stride=2))
model.add(Conv2D(32, (2,2), activation='relu'))
model.add(MaxPooling2D(filter_size=(2,2),stride=2))
model.add(Flatten())
model.add(Dense(output_units=128, activation='relu'))
model.add(Dense(output_units=10, activation='identify'))

# 任意のオプティマイザと損失関数を設定してモデルをコンパイルする
model.compile(optimizer='adam', loss='categorical_crossentropy' \
    , metrics='accuracy', output_units=10, batch_size=128)

# モデルをトレーニングする
model.fit(x_train=x_train, x_test=x_test, t_train=t_train, t_test=t_test, epochs=5)

# テストデータで精度を確認する
model.evalute(x_test, t_test)
