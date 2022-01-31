# ゼロから作るDeepLearning輪読会 in KV \[Ver. 1.1.1\]

## ■概要
* [ゼロから作るDeep Learning ―Pythonで学ぶディープラーニングの理論と実装](https://www.amazon.co.jp/dp/4873117585)の輪読会で使用するソースコード管理用のリポジトリです。
* [サンプルコードのリポジトリ](https://github.com/oreilly-japan/deep-learning-from-scratch)をベースとして配置しています。変更点として`ch0X/`直下に配置してあったソースコード類を`ch0X/sample/`直下に移動しています。
* 公開リポジトリのためアップロード内容にはくれぐれもご注意ください。

## ■勉強会メンバー向け

### 発表用ソースコードの管理
* 発表で使用するソースコードはすべて「main」ブランチにプッシュしてください。
* `ch0X/presentation/` 直下にフォルダを作成してソースコードを配置してください。
  - sampleをそのまま使う場合もコピーして配置してください
  - フォルダ名は自身の担当範囲の`[最初のサブセクション]-[最後のサブセクション]`としてください。<br>（例：ch02/presentation/2.2.1-2.3.2/XXXXX.py)
* commonとdatasetはフォルダ分けせずpresentation直下に配置してください。
#### 補足
* 発表準備がしやすいように発表単位でフォルダを分ける運用にしています。
  - 他の方が作成したソースを流用する場合は自身の担当回用フォルダにコピーから使用してください。
  - 他の方が作成したをモジュールをインポートして使用する場合はコピー不要です。直接参照してください。
* 逆にcommonとdatasetは共有リソースにしたいため、このフォルダ内は同一ファイルを皆で更新していきます。（デグレードに注意してください）

### プッシュ時の注意
* 発表1時間前には使用するソースコードをプッシュしてください。
* プルリクエストは非推奨とします。（誰もレビューしないので承認作業が単なる手間になるため）
* エラーを含むソースコードはプッシュしないでください。
* デグレード防止のため競合を解決してからプッシュしてください。

### ブランチ管理
ブランチ作成方針を以下とします：
#### 種類
大きく2種類に分けて管理します。用途ごとに命名規則に従って作成してください。

■常時ブランチ<br>
ブランチ単位で独立運用するためのブランチ
| 用途 | ブランチ名 |
| - | - |
| 発表ソース管理 | main |
| 各メンバー用 | [メンバー名(英数字)]_WORK |

■一時ブランチ<br>
特定の課題解決に対して作成し、作業完了後削除するブランチ
| 用途 | ブランチ名 |
| - | - |
| 発表準備用　 | dev[X.X.X-X.X.X(サブセクション番号)] |
| 特定要件の開発用 | dev[開発要件名(英数字)] |

#### 注意
* 自分用ブランチの作成は原則1人1ブランチでお願いします。（どのように運用していただいても構いません）
* 他人のブランチへのプッシュはNGとします。
* 一時ブランチは用済みになった時点で削除お願いします。

#### 補足
ブランチ運用イメージ

![ブランチ運用例](https://user-images.githubusercontent.com/65288037/148639617-335913f9-77d5-4b63-972b-2d2d9f9b7005.png)

## ■聴講者向け

* 「main」ブランチに発表に使用するソースコードをまとめていきます。<br>
  クローン、ダウンロードの際は「main」ブランチを使用してください。
* それ以外のブランチは勉強会メンバーの個人の作業ブランチです。閲覧はご遠慮ください。<br>
　（作業用ブランチであり、他者に共有する目的でソース管理しているブランチではないため）
* 勉強会メンバー以外の方のプッシュを禁止します。（権限上出来ないはずですが、明示的に注意事項として記載します）
 
