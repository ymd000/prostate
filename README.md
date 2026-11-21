# prostate

前立腺の病理画像を5クラスに分類するU-Net segmentationのベースラインです。
症例単位の5-fold CVで学習し、全画像のOOF予測とDice・IoUを保存します。
旧実装は `__archive/` に保持しています。開発ルールは [AGENTS.md](AGENTS.md) を参照してください。

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
- 症例を均等に選び、その症例の画像からランダムに256×256のパッチを抽出。90度回転と反転で拡張。
- 既定値はbatch size 1、AdamW、学習率0.0001、20 epochs、1 epochあたり256パッチ。
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

`--epochs`、`--samples-per-epoch`、`--patch-size`、`--batch-size`、`--base-channels`、
`--halo`、`--seed` などを指定できます。`--no-amp` で混合精度を無効化します。
`--amp-dtype auto|bfloat16|float16` でAMP精度、`--normalization group|none` で
畳み込みブロックの正規化を指定できます。実際に使った精度は起動ログとcheckpointに記録します。
別の設定を試す場合は、新しい `--run-dir outputs/実験名` を指定してください。
分割や学習設定が既存の実験と異なる場合は、上書きせずエラーにします。
`--resume` は完了foldのスキップです。未完了foldは最初から学習し直します。
同じ実験へのコマンドは順に実行してください。

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

`unet-stable` は32画像・ファイル名上の23症例を対象に、seed 42、各fold 20 epochs、
1 epochあたり256パッチ、GroupNorm、学習率0.0001、BF16で実行しています。
症例平均のforeground macro Diceは **0.1701**、IoUは **0.1180** でした。
クラス別の症例平均Diceは、非腺領域0.8935、非がん0.4108、GP3 0.0787、GP4 0.1678、GP5 0です。
全foldでGP5の予測画素数は0でした。これらは開発中のベースライン評価です。

同じseedとパッチ抽出処理を再現した集計では、全foldの全20 epochsで全5クラスが
学習パッチに含まれていました。各foldの学習パッチは合計5,120枚で、GP5を1画素以上含むものは
1,089〜1,456枚、GP5が有効画素の1%以上を占めるものも628〜969枚ありました。
学習画素中のGP5の割合は0.70〜1.54%です。検証側にも全foldで全5クラスが存在します。
GP5の成績が0なのは、fold内にそのクラスが存在しなかったためではありません。

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
    │   ├── history.csv     # epochごとのtrain lossと秒数
    │   └── evaluation.json # そのfoldのOOF画像別評価
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
混同行列をまとめてDice・IoUを計算し、症例を等しい重みで平均します。
背景（ID 0）を除いた4クラスのmacro平均と、全5クラスの個別指標を記録します。
正解・予測の両方に存在しないクラスは `null` として平均から除きます。

画像平均、foldごとの症例平均とその平均・標準偏差、全画素をまとめた評価も別に保存します。
画像サイズが違うため、全画素の評価と症例平均は異なります。
一部のfoldだけが完了している場合は `complete: false` になり、全OOFの結果とは区別されます。
OOFを見ながら設定を選んだ場合、その値は開発用CV評価として扱ってください。

## コードとテスト

`prostate/` は `config.py`、`data.py`、`folds.py`、`model.py`、`train.py`、
`inference.py`、`metrics.py`、`pipeline.py` とCLIの `__main__.py` に分けています。
学習と推論のAMP精度の選択は `precision.py` にまとめています。

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
