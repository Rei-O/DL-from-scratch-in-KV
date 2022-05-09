from reiras.models import NuralNet
from reiras.layers import *
sys.path.append(os.path.join(os.path.join(os.path.dirname(__file__), '..'), '..'))  # 親ディレクトリの親ディレクトリのファイルをインポートするための設定
from dataset.presentation.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

model = NuralNet()
model.add(Input(x_train.shape[1]))
model.add(Dense(output_units=10, activation='relu'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy', output_units=10, batch_size=64)
model.fit(x_train=x_train, t_train=t_train, x_test=x_test, t_test=t_test, epochs=100)
y = model.predict(x_test)
print(y)
print(model.train_loss_list[-1])
print(model.train_evalute_list[-1])
print(model.test_evalute_list[-1])