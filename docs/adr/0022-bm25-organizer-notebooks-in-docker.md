# ADR-0022: BM25 ベースラインは組織者のノートブックを当時のバージョンで固定した Docker で実行する

- Status: Accepted
- Date: 2026-09-27
- ADR-0012 の「指定構成で再現する」を実行環境の面で具体化する

## 決定

BM25 ベースライン (ADR-0012) は、組織者リポジトリ
[ntcirtransfer/transfer1](https://github.com/ntcirtransfer/transfer1) のノートブックを
**書き換えずにそのまま実行**して得る。実行環境は本体 (`Dockerfile`) とは別のイメージ
`bm25/Dockerfile` とし、ノートブックが動いた 2023 年時点のバージョンに固定する。

```
Python            3.7          ノートブックの kernel (language_info 3.7.12)
Java              11           ノートブックの Requirement
PyTerrier         0.9.2        Terrier 5.7 を読み込む
pyjnius           1.4.2        PyTerrier が JVM を呼ぶための橋渡し
SudachiPy         0.5.4        論文 (EMTCIR 2024) と ADR-0012 の指定
sudachidict_core  20230110     ノートブック実行時期に最も近い辞書
ir_datasets       0.5.5        組織者の dataset 定義が使う
組織者リポジトリ   9f875b68     archived 直前の main (2023-07-21)
```

### 実行の流れ

```
1. 配布物の MLIR.TGZ と TOPICS.TGZ を、読み取り専用のマウントから
   書き込めるディレクトリ (testcollections/ntcir/NTCIR-1/) に写す
2. preprocess-transfer1-train.ipynb を実行する
   tarball の展開、EUC-JP → UTF-8、JSON Lines と qrels TSV の生成
3. experiment-transfer1-train.ipynb を実行する
   SudachiPy でトークナイズ、Terrier でインデックス、BM25 で上位 1,000 件を検索
4. run ファイル runs/ntcir17-transfer/train/MyRun-BM25.res.gz が残る
```

ノートブックは papermill でヘッドレスに実行し、出力つきのノートブックも保存する。

ノートブックに加える変更は 1 箇所だけとする。`JAVA_HOME = 'FIT YOUR ENVIRONMENT'` という
プレースホルダを、イメージ内の JDK の場所に置き換える。イメージのビルド時に `sed` で
置き換え、ノートブックのファイル自体はリポジトリに持ち込まない。

### 配置

| コンテナ内 | ホスト | 中身 |
|---|---|---|
| `/data/ntcir` (読み取り専用) | `${NTCIR_DATA_DIR}` | 配布物 |
| `/work/transfer1/testcollections` | `cache/bm25/testcollections` | 展開と変換の結果 |
| `/work/transfer1/indexes` | `cache/bm25/indexes` | Terrier のインデックス |
| `/work/ir_datasets` | `cache/bm25/ir_datasets` | ir_datasets のキャッシュ |
| `/work/transfer1/runs` | `results/bm25` | run ファイルと実行済みノートブック |

いずれも `.gitignore` の対象で、コミットされない。run ファイルには文書 ID しか入らないが、
展開した文書本文が `cache/bm25/testcollections` に残るため、この扱いは配布物と同じにする。

### 再現の確認

ノートブックに保存されている出力には、`pt.Experiment` が表示した nDCG の全トピック平均
`0.526288` が残っている。この値は ir_measures の既定 (qrels の値 A=2, B=1 をそのまま利得に
使う段階的 nDCG) で計算されたもので、Transfer-1 の公式の評価 (2 値、ADR-0011) とは定義が
違う。したがって層の結果と並べる用途には使わず、**実行環境と手順が組織者と揃ったことの
確認**にだけ使う。実行後に同じ値が表示されなければ、バージョンかデータの違いを疑う。

run ファイルを層の結果と同じ定義で評価する処理 (run を読み、`evaluate()` に渡す) は
この ADR の範囲外とし、別に決める。

## 検討した選択肢

- **本体の環境 (Python 3.13) で PyTerrier と SudachiPy を使って自前で組む。**
  SudachiPy 0.5.4 は Python 3.13 でビルドできず (Cython のコンパイルエラー)、現行の 0.7.0 と
  現行の辞書を使うことになる。トークナイズが組織者と一致する保証がなくなり、ADR-0012 の
  「実装差で数値が動く」懸念がそのまま残る
- **本体の Docker イメージに JDK を足す。** SudachiPy の問題は解決しない。本体の実験に
  JVM は要らず、BM25 のためだけにイメージを重くする理由がない
- **ノートブックに保存されている数値をそのまま使う。** 全トピック平均の 1 つの数字しか
  なく、しかも段階的利得で計算されている。トピックごとの分析にも層との比較にも使えない
- **ノートブックを Python スクリプトに書き直して実行する。** 書き直した時点で
  「そのまま再現した」と言えなくなる。papermill を使えば書き直さずに済む
- **組織者リポジトリをサブモジュールにする。** 変更しないものをリポジトリの履歴に
  結びつける必要がない。ビルド時にコミットを固定して clone すれば足りる

## 結果

- Python 3.7 は EOL であり、イメージのベースは `python:3.7-slim-bullseye` に固定される。
  Debian bullseye には `openjdk-11-jdk-headless` があり、JDK の場所
  `/usr/lib/jvm/java-11-openjdk-amd64` はノートブックの保存出力にある組織者の環境と同じになる
- pyjnius は最新版の sdist が Cython 3.1 (Python 3.8 以上) を要求するため、wheel のある
  1.4.2 に固定しないと PyTerrier 0.9.2 が入らない
- イメージのビルドにネットワークが要る (pip、組織者リポジトリの clone、Terrier の jar の
  取得)。jar はビルド時に `pt.init()` を一度呼んで取り込み、実行時には要らないようにする
- 実行時にノートブック内の `pip install` セルが動くが、`PIP_NO_INDEX=1` で index を
  引かせず、固定したバージョンが置き換わらないようにする
- インデックス作成に 5 分ほどかかる (ノートブックの保存出力では 4 分 46 秒)。run ファイルが
  既にあるときは実行しない。作り直すときは明示する
