# ゼロから作るDeepLearning輪読会 in KV

## ■概要
* [ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装](https://www.amazon.co.jp/dp/4873117585)の輪読会で使用するソースコード管理用のリポジトリです。
* [サンプルコードのリポジトリ](https://github.com/oreilly-japan/deep-learning-from-scratch)をベースとして配置しています。変更点として`ch0X/`直下に配置してあったソースコード類を`ch0X/sample/`直下に移動しています。
* 公開リポジトリのためアップロード内容にはくれぐれもご注意ください。

## ■運用ルール

### 発表用ソースコードの管理
* 発表で使用するソースコードはすべて「main」ブランチにプッシュしてください。
* `ch0X/presentation/` 直下にフォルダを作成してソースコードを配置してください。
  - sampleをそのまま使う場合もコピーして配置してください
  - フォルダ名は自身の担当範囲の`[最初のサブセクション]-[最後のサブセクション]`としてください。<br>（例：ch02/presentation/2.2.1-2.3.2/XXXXX.py)
* commonとdatasetはフォルダ分けせずpresentation直下に配置してください。
#### 補足
発表準備がしやすいように発表単位でフォルダを分ける運用にしています。<br> 他の方が作成したソースを流用する場合もコピーしてください。
逆にcommonとdatasetは共有リソースにしたいため、このフォルダ内は同一ファイルを皆で更新していきます。（デグレードに注意してください）

### プッシュ時の注意
* 発表1時間前には使用するソースコードをプッシュしてください。
* プルリクエストは非推奨とします。（誰もレビューしないので承認作業が単なる手間になるため）
* エラーを含むソースコードはプッシュしないでください。
* 競合を解決してからプッシュしてください。（デグレード防止のため）

### ブランチ管理
* 自分用ブランチの作成は自由です。（どのように運用していただいても構いません）
* 他人のブランチへのプッシュはNGとします。（閲覧は自由です）
* 原則1人1ブランチでお願いします。
