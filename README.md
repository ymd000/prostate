# prostate

前立腺の病理画像を5クラスに分類するU-Net segmentationのベースラインです。
症例単位の5-fold CVで学習し、全画像のOOF予測とDice・IoUを保存します。
旧実装は `__archive/` に保持しています。開発ルールは [AGENTS.md](AGENTS.md) を参照してください。

## 当面の目標（2026-11-22）

5クラスsegmentationのIoU改善を目指し、特に背景（non_gland）以外の
non_cancer・GP3・GP4・GP5を優先します。主指標は既存の
`primary_case_mean.foreground_macro_iou`（背景を除く4クラスの症例平均）です。
全5クラスの個別IoU・foldごとの成績を併記し、固定した症例単位5-foldのOOFで比較します。
GG関連の方針は病理医のレビュー後に検討します。

## 環境

- Python 3.12
- [uv](https://docs.astral.sh/uv/) による依存管理
- PyTorch・torchvision（LinuxではCUDA 13.2版、WindowsではCPU版）
- NumPy・SciPy・Pillow・OpenCV・Matplotlib・tqdm
- 開発用: Ruff・pytest

依存バージョンは `uv.lock` で固定します。OpenCVはGUIを必要としない `opencv-python-headless` を利用します。
PyTorchの配布元は [uv公式の設定例](https://docs.astral.sh/uv/guides/integration/pytorch/) に基づいて設定しています。
LinuxではRTX 3050・現在のドライバに合わせてCUDA 13.2版を指定しています。
別の環境ではGPUとドライバに合わせて配布元を変更してください。

## セットアップ

uvをインストールしたうえで、リポジトリのルートで実行します。

```bash
uv sync --locked
```

`.venv/` に仮想環境が作成され、開発用ツールもインストールされます。

```bash
# 環境の確認
uv run python --version
uv run python -c "import cv2, numpy, torch, torchvision; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

# 静的チェックとフォーマットの確認（旧実装は対象外）
uv run ruff check .
uv run ruff format --check .
```

スクリプトは `uv run python path/to/script.py` で実行します。
仮想環境を直接有効化する場合は `source .venv/bin/activate` を実行してください。

## 依存の追加

```bash
uv add パッケージ名
uv add --dev 開発用パッケージ名
```

`pyproject.toml` と `uv.lock` を一緒にバージョン管理します。
データは `data/`、学習済み重みは `weights/`、生成物は `outputs/` に配置するとGitの管理対象から除外されます。

## 外付けドライブと保存先

現在の配置は次のとおりです。リンクとデータはGit管理から除外します。

```text
data/completed -> /run/media/ymd/KIOXIA/Prostate/completed
outputs        -> /run/media/ymd/KIOXIA/Prostate/outputs
```

別の環境では、リポジトリ直下でリンクを作成します。

```bash
mkdir -p data
ln -s /path/to/Prostate/completed data/completed
ln -s /path/to/Prostate/outputs outputs
```

`data/completed/` 直下の同名JPEG・PNGをペアにします。現在は32組・ファイル名上の23症例です。
`15-0459_5_1` の場合、最初の `_` より前の `15-0459` を症例IDとして扱います。
同一患者に複数の症例IDがある場合は、分割前にグループの定義を見直してください。
`annotation.csv` がない画像も、PNGがあれば対象です。XCF・サムネイル・PPTXは読み込みません。
入力データは変更せず、キャッシュや予測結果を `outputs/` に保存します。
外付けドライブはリンク先と同じパスにマウントして使います。

## ベースラインの仕様

- 4段のダウンサンプリングとskip connectionを持つU-Net。初期チャネル数32、事前学習なし。
  各3×3畳み込みの後にGroupNormを入れ、batch size 1でも特徴のスケールを整える。
- 入力はRGBを0〜1に変換。出力は5クラスのlogitsで、CrossEntropyLossで学習。
- 訓練画像を256×256の重ならないグリッドに分け、各epochで全パッチを1回ずつ処理する。
  毎epochで順序をshuffleし、90度回転と反転で拡張。端の画像外領域はlossから除外する。
- 既定値はbatch size 1、AdamW、学習率0.0001、20 epochs。1 epochのパッチ数はfoldごとに決まる。
- CUDA利用時はAMPを有効化。対応GPUではBF16を優先し、それ以外ではFP16を使う。
  lossはFP32で計算し、GradScalerはFP16時のみ使う。CPUも利用可能。
- 症例数を優先して均等化した、seed固定の5-fold。画像枚数を同数にする分割や、クラスの層化は行わない。
- foldごとにモデルを初期化し、固定epoch数で学習。OOFの正解をepoch選択・early stoppingに使わない。
- OOF推論では256×256のタイルに周囲32ピクセルの文脈を持たせ、中央192×192の領域をつなぐ。
  端は反射パディングし、画像全体を元の縦横サイズで評価する。

### PNGのクラス変換

元PNGは、組織画像に色を塗ったアノテーションです。旧実装と同じく各RGBチャネルの
`> 0` を判定し、次のIDへ変換します。透明部分は0、全チャネルが正の組織画素も0です。

| ID | クラス | 元PNGの色 |
|---|---|---|
| 0 | non_gland | 透明、白、RGB全チャネルが正の組織画素 |
| 1 | non_cancer | 黒、紫 |
| 2 | GP3 | 青、シアン |
| 3 | GP4 | 緑、黄 |
| 4 | GP5 | 赤 |

教師マスクはこの変換を使うPNGが前提です。出力する予測PNGはパレット形式で、
画素値そのものが0〜4のクラスIDです。表示色は0=透明、1=黒、2=青、3=緑、4=赤です。

## 実行

すべてリポジトリ直下で実行します。

```bash
# 全画像を検査・キャッシュし、症例単位のfold割当を保存（学習はしない）
uv run python -m prostate prepare

# 全5foldの学習、OOF予測、評価
uv run python -m prostate cv --device cuda

# foldを一つずつ実行する場合
uv run python -m prostate cv --device cuda --fold 0

# 完了したfoldをスキップして続行
uv run python -m prostate cv --device cuda --resume

# 保存済みOOF配列から評価を再計算
uv run python -m prostate evaluate

# 1画像をfoldモデルで推論（OOF用途では未学習の症例を指定する）
uv run python -m prostate predict data/completed/15-1428_5_2_validation.jpg \
  --checkpoint outputs/unet-baseline/fold_0/checkpoint.pt \
  --output outputs/prediction.png --device cuda

# オプション一覧
uv run python -m prostate cv --help
```

`--device auto`（既定）はCUDAが利用可能ならGPU、それ以外はCPUを使います。
LinuxのPyTorch取得先は `https://download.pytorch.org/whl/cu132` に固定しています。
CPU版からの切り替えや環境の再作成には `uv sync --locked` を実行します。
`--device cuda` はCUDAが使えなければエラーで終了します。
`nvidia-smi` のCUDA表記だけでは、PyTorchでGPUが使えることは確認できません。

GPU版一式は約3GBのダウンロードになります。大きなファイルの取得が繰り返される場合は、
タイムアウトと同時取得数を調整して再試行できます。今回の取得済みキャッシュは
`tmp/uv-cache` にあり、同じ場所を指定すると再利用します。

```bash
UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 \
  uv sync --locked --cache-dir tmp/uv-cache
```

`--epochs`、`--patch-size`、`--batch-size`、`--base-channels`、
`--halo`、`--seed` などを指定できます。`--no-amp` で混合精度を無効化します。
`--amp-dtype auto|bfloat16|float16` でAMP精度、`--normalization group|none` で
畳み込みブロックの正規化を指定できます。実際に使った精度は起動ログとcheckpointに記録します。
別の設定を試す場合は、新しい `--run-dir outputs/実験名` を指定してください。
分割や学習設定が既存の実験と異なる場合は、上書きせずエラーにします。
`--resume` は完了foldのスキップです。未完了foldは最初から学習し直します。
同じ実験へのコマンドは順に実行してください。

### 凍結GigaPath-Flashと学習するdecoder

`--encoder gigapath-flash` で、凍結したGigaPath-Flashの空間tokenを使う構成に切り替えます。
既定では256×256のRGBタイルをImageNetのmean/stdで正規化し、ViT-S/16からCLSを除いた
16×16×384の特徴を取り出します。`TokenDecoder` はこの特徴を4段で拡大し、
5×256×256のlogitsを出力します。encoderとのskip connectionはなく、学習するのはdecoderです。

```bash
# 初回のみ、全画像の空間tokenを抽出（学習はしない）
uv run python -m prostate extract --encoder gigapath-flash --device cuda \
  --extract-batch-size 1

# 保存したtokenを使って各foldのdecoderを学習
uv run python -m prostate cv --encoder gigapath-flash --device cuda \
  --run-dir outputs/gigapath-full-grid
```

特徴は既定で `outputs/tokens/<画像ID>.h5` にfloat16で保存します。
各ファイルにはパッチの位置・元画像サイズ・patchサイズ・encoder名も記録します。
学習時はこの特徴を読み、OOF・単画像推論ではRGBタイルに凍結encoderを適用します。
checkpointにはdecoderの重みを保存し、推論時にGigaPath-Flashを読み込みます。
`predict` はcheckpointのモデル構成に応じてU-NetとGigaPath版を切り替えます。

特徴抽出では既存のh5を再利用します。入力画像やpatchサイズを変える場合は、
別の `--token-dir` を指定して再抽出し、学習時にも同じ保存先を指定してください。

### 大きい入力での比較（1024px）

現在利用している `wsi-toolbox` のGigaPath-Flashは `dynamic_img_size=True` です。
`--patch-size` は元画像から切り出してencoderに渡す正方形の一辺で、
ViT内部の `patch_size=16` は変わりません。画像を縮小せずに入力し、
位置埋め込みだけをtokenグリッドのサイズに合わせて補間します。
元画像の細胞の画素サイズを保ちつつ、広い範囲を一度に処理できます。

| `--patch-size` | 空間tokenグリッド | 空間token数（CLS除く） |
|---|---|---:|
| 256 | 16×16 | 256 |
| 512 | 32×32 | 1,024 |
| 768 | 48×48 | 2,304 |
| 1024 | 64×64 | 4,096 |

encoderの系列にはCLSがさらに1個加わります。現在の特徴保存・読込とdecoderは
正方形・16の倍数の入力サイズに対応します。長方形の元画像全体を一度に渡す処理はありません。
入力を大きくするとattention部分の演算量はtoken数の二乗に応じて増え、
decoderの学習メモリも増えます。抽出batchと学習batchはまず1で実行します。

```bash
# 1024pxグリッドの特徴を別の保存先に抽出
uv run python -m prostate extract --encoder gigapath-flash --device cuda \
  --patch-size 1024 --token-dir outputs/tokens-1024 --extract-batch-size 1

# 同じ1024px入力で全5foldを学習・OOF評価
uv run python -m prostate cv --encoder gigapath-flash --device cuda \
  --patch-size 1024 --token-dir outputs/tokens-1024 \
  --run-dir outputs/gigapath-1024-weighted-ce --class-weighting inverse-sqrt \
  --epochs 20 --seed 42 --n-splits 5 --batch-size 1 --halo 32
```

比較対象は256px・重み付きCEの `outputs/gigapath-weighted-ce` とします。
同じ画像・症例単位5-fold、seed 42、20 epochs、全グリッド巡回、inverse-sqrt重み付きCE、
学習率0.0001、batch size 1、halo 32で比較します。別環境でも同じデータを配置し、
両実験の `folds.csv` の割当が一致することを確認してください。
パッチが大きいほど1epochのパッチ数・更新回数は減るため、各foldの
`patches_per_epoch`・`batches_per_epoch` も併記します。
OOFと最終checkpointの訓練側評価も、その実験の入力サイズを使います。
主指標は症例平均foreground macro IoUとし、全5クラスのIoUとfold別成績を併記します。

CPUでは、事前学習済みencoderと未学習decoderに人工入力を渡し、
256・512・768・1024pxで同じ縦横サイズの5クラスlogitsが出て、値が有限であることを確認しました。
この確認には学習・精度比較・GPUメモリ測定を含みません。

### 再現実験：validationの1症例をholdoutに固定

`holdout` コマンドは、`completed/` にある `15-1428_5_2_validation` の症例ID
`15-1428` を検証用に固定し、残りの症例でモデルを1つ学習します。
現在の32画像・23症例では、訓練31画像・22症例、holdout 1画像・1症例になります。
対象は `completed/` のJPEG・PNGペアで、外付けドライブの旧 `train/` は使いません。
同じ症例IDの別画像が追加された場合も、その症例の全画像をholdoutに割り当てます。
`--holdout-case-id` の既定値は `15-1428` で、別の症例IDも指定できます。

これは旧実装を参考にした分割条件の再現です。学習は現行の全グリッド巡回・固定epochを使い、
holdoutをcheckpointやepoch数の選択に使いません。lossの重みは訓練31画像だけから計算します。
入力サイズなどの既定値はCVと共通で、U-Net・GigaPath-Flashの両方を利用できます。

```bash
# U-Netの256px入力でholdout実験
uv run python -m prostate holdout --device cuda \
  --run-dir outputs/unet-holdout-15-1428

# GigaPath-Flashの1024px特徴を抽出（抽出済みなら再利用）
uv run python -m prostate extract --encoder gigapath-flash --device cuda \
  --patch-size 1024 --token-dir outputs/tokens-1024 --extract-batch-size 1

# GigaPath-Flashの1024px入力・重み付きCEでholdout実験
uv run python -m prostate holdout --encoder gigapath-flash --device cuda \
  --holdout-case-id 15-1428 --patch-size 1024 --token-dir outputs/tokens-1024 \
  --run-dir outputs/gigapath-1024-holdout-15-1428 --class-weighting inverse-sqrt \
  --epochs 20 --seed 42 --batch-size 1 --halo 32
```

分割と設定だけを保存する場合は、実行予定の `holdout` コマンドに `--prepare-only` を加えます。
画像・ラベルのキャッシュを準備し、encoderの読込・特徴抽出・学習は実行しません。
同じ設定のコマンドから `--prepare-only` を外せば学習を開始できます。
完了した実験の再利用には `--resume` を加えます。未完了の学習は最初から実行し直します。
入力サイズやlossなどを変える場合は、新しい `--run-dir` を指定します。

保存先には `config.json`、`manifest.json`、画像ごとのtrain/holdout割当を記した `split.csv`、
`holdout/<画像ID>.npy`・`.png`、`holdout_images.json`、`holdout_metrics.json` を作成します。
モデル・loss設定・訓練履歴・訓練側評価は `fold_0/` に保存します。この実験で学習するモデルは1つです。
訓練側との比較は `train_metrics.json` の `holdout_primary_case_mean` などに保存します。
`evaluate --run-dir ...` と `evaluate-train --run-dir ...` で再集計・訓練側の再評価もできます。
holdoutの指標はこの1症例の結果として扱い、5-fold OOFの結果と分けて記録します。

### 全パッチを1epochで巡回する学習

U-Netと凍結GigaPath-Flashの両方で、訓練用の全グリッドパッチを各epochで1回ずつ使います。
右端・下端も含め、RGBは反射padding、画像外の正解ラベルはIGNORE_INDEX（255）で埋めます。
最後のbatchが小さくても捨てません。シャッフルと拡張はseedで再現できます。
全パッチを一巡するので、大きな画像はパッチ数に応じて多く使われます。

旧方式の `--samples-per-epoch` は廃止しました。`config.json` に
`patch_sampling: "full_grid_v1"`、checkpointに `patches_per_epoch` と `batches_per_epoch`、
`history.csv` に各epochの実際の処理パッチ数・batch数を保存します。
旧ランダム抽出の実験とは設定が異なるため、新しい保存先で開始してください。

```bash
# 既存のtokensを使い、decoderを全パッチ巡回で学習
uv run python -m prostate cv --encoder gigapath-flash --device cuda \
  --run-dir outputs/gigapath-full-grid

# U-Netも同じ全パッチ巡回
uv run python -m prostate cv --device cuda --run-dir outputs/unet-full-grid
```

現在の32画像・同じfold分割・256pxグリッドでは、fold 0〜4の1epochあたりの
訓練パッチ数は順に3,368、3,282、1,923、3,393、3,630です。
20 epochsは全パッチ20巡となり、旧方式の5,120パッチ/foldより更新回数が増えます。
既存のtokenキャッシュは再利用できます。サイズや位置が完全なグリッドと一致しないキャッシュは
学習前にエラーにします。fold分割、OOF推論・評価の方法は同じです。

### 重み付きCrossEntropyの比較

`--class-weighting inverse-sqrt` で、各foldの訓練画像だけから求めた画素頻度に応じて
CrossEntropyのクラス重みを付けます。既定の `none` は従来どおりの重みなしCEです。
U-Net・GigaPath-Flashの両方に適用できます。

各クラスの訓練画素数を `n_c`、訓練総画素数を `N` とすると、重みは
`1 / sqrt(n_c / N)` に比例します。存在するクラスの重みの平均が1になるよう正規化し、
訓練側に存在しないクラスの重みは0とします。画素数には画像外のpaddingを含めません。
OOF側のラベル・画素数や診断CSVは重み計算に使いません。

```bash
# 既存tokenを使い、重み付きCEを同じ20 epochs・全パッチ巡回で比較
uv run python -m prostate cv --encoder gigapath-flash --device cuda \
  --class-weighting inverse-sqrt --run-dir outputs/gigapath-weighted-ce
```

設定の `class_weighting` は `config.json` に保存します。各foldの学習開始時に
`fold_N/loss.json` へ訓練画像・症例ID、クラス別画素数・頻度、実際に使う重み、
計算式、IGNORE_INDEX、FP32のloss計算条件を保存します。同じ情報はcheckpointの `loss` にも入ります。
重み付きCEもlossはFP32で計算し、画像外のIGNORE_INDEX（255）を除外します。

重みが変わると訓練lossの意味も変わるため、比較はOOFの
`primary_case_mean.foreground_macro_iou` と各クラスのIoUを主に使います。
評価指標の計算方法は同じで、lossのクラス重みは評価には適用しません。
重みなし実験と重み付き実験は別の `--run-dir` に保存してください。
重み設定の記録がない従来の実験は、重みなしの条件でのみ `--resume` できます。

### 訓練側のクラス別指標と履歴

U-Net・GigaPath-Flashの両方で、毎epochの `history.csv` に
`online_dice_GP5`、`online_precision_GP5`、`online_recall_GP5` など、全5クラスの
Dice・IoU・precision・recallを保存します。`history_metrics.jsonl` には各epochの混同行列、
正解・予測・正解一致の画素数も保存します。画像外のIGNORE_INDEXは集計から除外します。

この履歴は `online_training_batches_v1` という集計方法です。学習中のforward結果を使うため、
epochの途中で重みが更新され、回転・反転も含みます。既存のforwardから集計するので、
毎epochの評価用forwardは追加しません。履歴の値は、そのepoch終了時の固定モデルの成績とは異なります。

各foldの学習・OOF推論が終わった後には、最終checkpointの訓練画像を `eval()`・拡張なしで
評価します。OOFと同じ入力サイズ（既定256px）・halo付きタイル結合を使い、訓練画像の全画素を評価します。
この推論時間は追加されます。保存先は次のとおりです。

- `fold_N/train_images.json`：訓練画像ごとの混同行列・クラス別指標。
- `fold_N/train_metrics.json`：訓練側の症例平均・画像平均・全画素集計、checkpoint情報、推論条件。
- 実験直下の `train_metrics.json`：foldごとの訓練側とOOFの指標、前景macro Diceの差。

同じ画像が複数foldの訓練側に登場するため、訓練側の結果はfoldごとに表示します。
症例平均と全画素集計は分けて記録します。precisionは予測画素が0なら `null`、recallは
正解画素が0なら `null` で、平均から除外します（CSVでは空欄）。Dice・IoUは両方に
そのクラスがない場合だけ `null` です。訓練画像の予測配列は1画像分ずつ一時保存し、集計後に削除します。

既存モデルでも、再学習せずに最終checkpointの訓練側評価を追加できます。

```bash
# 全checkpointの訓練側を評価。旧ランダム抽出の実験でも利用可能
uv run python -m prostate evaluate-train --run-dir outputs/gigapath --device cuda

# foldを指定して確認。--fold は複数回指定可能
uv run python -m prostate evaluate-train --run-dir outputs/gigapath --device cuda --fold 0

# 評価レポートを別の場所に保存する場合
uv run python -m prostate evaluate-train --run-dir outputs/gigapath --device cuda \
  --output-dir outputs/gigapath/training-diagnostics
```

このコマンドは既存の重み・履歴・OOF結果を変更しません。過去のepochごとのクラス別指標は
復元せず、最終モデルの評価を保存します。部分的な評価では `complete: false` と未評価foldを記録します。
今後のCVでは自動実行し、`cv --resume` でも訓練側レポートが欠けている完了foldは
既存checkpointから補完します。checkpointが変わった場合は、古い訓練側レポートを再利用しません。
OOFをepoch選択やearly stoppingに使う処理は追加していません。

### 学習lossが非有限値になった場合

初期版（正規化なし・学習率0.001・FP16固定）では、実データのepoch 5で
lossが急増し非有限値になる事例がありました。原因の切り分けは未完了ですが、
既定値をGroupNorm・学習率0.0001・BF16優先に変更しています。
正規化とAMPの設定も `config.json` に保存し、旧条件の実験との混在を防ぎます。
旧checkpointの単画像推論は引き続き利用できます。

既存のログを残して新しい条件で開始する例です。

```bash
uv run python -m prostate cv --device cuda --run-dir outputs/unet-stable
```

AMPの影響を切り分ける場合は、別の実験名と `--no-amp` を指定します。
非有限lossでは処理を停止し、fold・epoch・batch・精度・logitsの有限性を表示します。
BF16／FP32で勾配が非有限値になった場合も、パラメータを更新せず停止します。
修正後の既定条件では、RTX 3050 Laptop GPU（4GB）で全5foldが完走しました。

### 初回OOF評価の記録

旧ランダム抽出方式の `unet-stable` は32画像・ファイル名上の23症例を対象に、seed 42、各fold 20 epochs、
1 epochあたり256パッチ、GroupNorm、学習率0.0001、BF16で実行しています。
症例平均のforeground macro Diceは **0.1701**、IoUは **0.1180** でした。
クラス別の症例平均Diceは、非腺領域0.8935、非がん0.4108、GP3 0.0787、GP4 0.1678、GP5 0です。
全foldでGP5の予測画素数は0でした。これらは開発中のベースライン評価です。

同じseedとパッチ抽出処理を再現した集計では、全foldの全20 epochsで全5クラスが
学習パッチに含まれていました。各foldの学習パッチは合計5,120枚で、GP5を1画素以上含むものは
1,089〜1,456枚、GP5が有効画素の1%以上を占めるものも628〜969枚ありました。
学習画素中のGP5の割合は0.70〜1.54%です。検証側にも全foldで全5クラスが存在します。
GP5の成績が0なのは、fold内にそのクラスが存在しなかったためではありません。

### GigaPath-Flashの重み付きCE評価（2026-11-22）

`outputs/gigapath`（通常CE）と `outputs/gigapath-weighted-ce`（inverse-sqrt重み付きCE）を比較しました。
どちらも凍結GigaPath-Flash＋TokenDecoder、32画像・23症例、seed 42の同じ症例単位5-fold、
20 epochs・全パッチ巡回です。manifestとfold割当は一致し、lossの重み設定を変更しています。

| 症例平均IoU | 通常CE | 重み付きCE |
|---|---:|---:|
| non_gland | 0.8839 | 0.8503 |
| non_cancer | 0.5380 | 0.5332 |
| GP3 | 0.3514 | 0.3682 |
| GP4 | 0.2406 | 0.2592 |
| GP5 | 0.0032 | 0.0467 |
| foreground macro IoU | 0.2890 | 0.3004 |

主指標の非背景macro IoUは5fold中4foldで改善しました。背景・非癌のIoUは低下しています。
GP5は全画素集計の再現率が1.86%→32.48%、適合率が13.73%→21.33%に上がりましたが、
誤検出は依然として多く残っています。これらは開発中のCV評価です。
lossの計算とtensor shapeを更新した構成図は
`outputs/tmp/architecture_gigapath_flash_weighted_ce.jpg` に保存しています（Git管理外）。

### 保存されるもの

```text
outputs/
├── cache/<画像ID>/<入力ファイルのfingerprint>/
│   ├── rgb.npy             # RGBのuint8配列（mmapでパッチを読み出す）
│   ├── labels.npy          # 正解クラスIDのuint8配列
│   └── sample.json         # 元ファイルのパス・サイズ・mtime、クラス画素数
└── unet-baseline/          # --run-dir で変更可能
    ├── config.json         # seed、epoch数、パッチサイズ、AMPなど
    ├── manifest.json       # 対象画像・症例ID、入力情報、キャッシュ先
    ├── folds.csv           # 画像ごとのfold、サイズ、クラス画素数
    ├── fold_0/ ... fold_4/
    │   ├── checkpoint.pt   # モデル重み、構造、設定、学習症例ID
    │   ├── loss.json       # CE設定、訓練側クラス画素数・頻度・重み
    │   ├── history.csv     # epochごとのloss、秒数、パッチ数、batch数、onlineクラス別指標
    │   ├── history_metrics.jsonl # epochごとの混同行列・画素数・online指標
    │   ├── train_images.json  # 最終checkpointの訓練画像別指標
    │   ├── train_metrics.json # 最終checkpointの訓練側集計・推論条件
    │   └── evaluation.json # そのfoldのOOF画像別評価
    ├── train_metrics.json  # foldごとの訓練側とOOFの比較
    ├── oof/<画像ID>.png    # 元サイズの予測マスク（画素値は0〜4）
    ├── oof/<画像ID>.npy    # 同じ予測のuint8配列
    ├── oof_images.json     # OOF画像別のDice・IoU・混同行列
    └── oof_metrics.json    # OOF全体の評価と未完了画像一覧
```

キャッシュは入力パス・ファイルサイズ・mtime・ラベル変換バージョンで識別し、
元データが変わった場合は別のキャッシュを作ります。RGB・ラベルのキャッシュは
合計で概ね総画素数×4バイトの容量を使います。全画像をまとめてRAMやVRAMに載せません。

### 評価の集計

主な指標は `oof_metrics.json` の `primary_case_mean` です。同じ症例の画像の
混同行列をまとめてDice・IoU・precision・recallを計算し、症例を等しい重みで平均します。
背景（ID 0）を除いた4クラスのmacro平均と、全5クラスの個別指標を記録します。
正解・予測の両方に存在しないクラスは `null` として平均から除きます。

画像平均、foldごとの症例平均とその平均・標準偏差、全画素をまとめた評価も別に保存します。
画像サイズが違うため、全画素の評価と症例平均は異なります。
一部のfoldだけが完了している場合は `complete: false` になり、全OOFの結果とは区別されます。
OOFを見ながら設定を選んだ場合、その値は開発用CV評価として扱ってください。

### OOFのサムネイル比較

保存済みのOOF結果から、定性的な確認用のJPEGを生成できます。
モデルの読み込み・学習・推論・評価指標の再計算は行わず、CPUで画像を描画します。
U-NetとGigaPath-Flashのどちらの実験にも利用できます。

```bash
# 全画像の比較カードと一覧。既定は前景macro Diceの低い順。
uv run python -m prostate thumbnails --run-dir outputs/gigapath-run

# 最初の8画像のみ、個別カードの画像の長辺を1024pxにする
uv run python -m prostate thumbnails --run-dir outputs/gigapath-run --limit 8 --max-side 1024

# 画像を指定して確認（--image-id は複数回指定可能）
uv run python -m prostate thumbnails --run-dir outputs/gigapath-run --image-id 14-0759_10_1
```

`--sort best|worst|id` で並び順、`--per-page` で一覧1枚あたりの画像数（既定8枚）、
`--output-dir` で保存先を変更できます。既定の保存先は実験フォルダ内の `thumbnails/` です。
個別カード内の各画像は長辺768px、一覧内では長辺320pxが既定です。
`--max-side` と `--contact-side` で変更できます。保存品質は `--jpeg-quality 95`、
色のにじみを抑えるため色差の間引きはなし（4:4:4）です。
表示と並び替えに使うDiceは各画像単位の値で、OOF全体の症例平均とは集計単位が異なります。

```text
outputs/gigapath-run/thumbnails/
├── sheet_001.jpg ...       # 元画像・正解・OOF予測・誤り分布の一覧
├── images/<画像ID>.jpg     # 元画像、正解、予測、正解overlay、予測overlay、誤り分布
└── index.json              # 今回生成した画像の一覧、fold、Dice、ページ名
```

表示色は非腺領域が薄い灰色、非がんが黒、GP3が青、GP4が緑、GP5が赤です。
overlayでは非腺領域を透明にして元画像を表示します。正解・予測マスクは最近傍で縮小します。
PNGは可逆圧縮なので、JPEGへの変更自体で細部は増えません。
微細な領域の確認には `--max-side` を大きくしてください。JPEGは閲覧用であり、
評価には保存済みのクラスID配列を使います。
誤り分布は元解像度で正解と予測が違う画素を数え、各表示領域内の誤分類率を
白（0%）から赤（100%）で表します。背景の誤分類も含みます。
元の画像・マスク・OOF配列・評価JSONは変更しません。

## コードとテスト

`prostate/` は `config.py`、`data.py`、`folds.py`、`model.py`、`train.py`、
`inference.py`、`metrics.py`、`pipeline.py` とCLIの `__main__.py` に分けています。
学習と推論のAMP精度の選択は `precision.py` にまとめています。
訓練foldからのクラス重み計算とCEの設定保存は `loss.py` で行います。
凍結encoderの特徴抽出は `extractor.py`、特徴からの訓練パッチ読込は `token_data.py`、
epochごとの指標記録は `training_history.py`、最終モデルの訓練側評価は `training_evaluation.py` です。
OOFの比較画像は `thumbnails.py` で生成します。

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

テストではラベル色変換、症例の分離、タイルの結合、集計単位の違い、
小さな6画像・5症例での学習から全5foldのOOF保存・再開・単画像推論までを確認します。
本学習の精度や速度を測るテストではありません。

## 旧実装

`__archive/` に、旧コード・Conda環境定義・README・ライセンスを内容を変更せず保存しています。
旧実装のCLI・重みとの互換性はありません。新しいパイプラインはPNGのクラス対応を引き継いでいます。
学習データ・学習済み重みはGitに含めていません。
