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
Python              3.7          ノートブックの kernel (language_info 3.7.12)
Java                11           ノートブックの Requirement
PyTerrier           0.9.2        ノートブックの指定
Terrier             5.7          ノートブックの保存出力 (built 2022-11-10)。helper 0.0.7
pyjnius             1.4.2        PyTerrier が JVM を呼ぶための橋渡し
SudachiPy           0.5.4        論文 (EMTCIR 2024) と ADR-0012 の指定
sudachidict_core    20230110     ノートブック実行時 (2023-02) の最新
ir_datasets         0.5.5        組織者の dataset 定義が使う
ir_measures         0.3.3        ノートブック末尾の pt.Experiment が使う
pytrec_eval_terrier 0.5.5        同上
組織者リポジトリ     9f875b68     archived 直前の main (2023-07-21)
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

ノートブックに加える変更は 2 箇所だけとする。いずれもイメージのビルド時に `sed` で
置き換え、ノートブックのファイル自体はリポジトリに持ち込まない。

1. `JAVA_HOME = 'FIT YOUR ENVIRONMENT'` というプレースホルダを、イメージ内の JDK の場所にする
2. `pt.init(tqdm='notebook')` に `version='5.7', helper_version='0.0.7'` を足す。渡さないと
   PyTerrier が最新の Terrier (5.11) を取り、組織者が使った 5.7 と食い違う。PyTerrier 0.9.2 は
   環境変数で版を指定できない

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

ノートブックに保存されている出力には、`pt.Experiment` が表示した nDCG の平均 `0.526288` が
残っている。この値には 2 つの注意がある。

- ir_measures の既定 (qrels の値 A=2, B=1 をそのまま利得に使う段階的 nDCG) で計算されており、
  Transfer-1 の公式の評価 (2 値、ADR-0011) とは定義が違う
- ノートブックが実行された 2023 年 2 月の `datasets/ntcir_transfer.py` は qrels として
  `rel2_ntc1-j1_0001-0030` (トピック 0001〜0030 だけ) を読んでいた。組織者はこれを
  2023-06-28 のコミット e50b5087 "Bug Fix: Wrong qrel files in the train set" で
  `0001-0083` に直したが、ノートブックは再実行されていない。つまり `0.526288` は
  **30 トピックの平均**である

したがって層の結果と並べる用途には使わず、**実行環境と手順が組織者と揃ったことの確認**にだけ
使う。確認は、得られた run をトピック 0001〜0030 の qrels で同じ定義で評価し、`0.526288` に
一致するかで行う。最初の実行では 6 桁まで一致した。修正後の qrels (83 トピック) では
`0.495412` になる。

```
トピック 0001-0030 (組織者と同じ条件)  nDCG = 0.526288
トピック 0001-0083 (修正後の qrels)    nDCG = 0.495412
```

run の先頭 (トピック 0001 の上位 10 件) は文書も順位も組織者の保存出力と同じで、スコアは
6 桁目以降だけ違う。

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
- ノートブックの `pip install` セルで入る pytrec_eval_terrier と ir_measures は、版を
  指定しないと Python 3.7 で動かない版 (0.5.10 / 0.4.1) が入り、末尾の `pt.Experiment` が
  失敗する。run はその前に保存されるが、失敗を残さないために 2022〜2023 年の版に固定する
- Debian bullseye は LTS が終わり通常のミラーから消えているため、apt は archive.debian.org を
  参照する。SudachiPy・zlib-state・pyautocorpus は cp37 の wheel が無く sdist からビルドする
  ので、build-essential と pcre・zlib のヘッダが要る
- コンテナはホストのユーザー ID で動かし、生成物を root 所有にしない。Makefile の `bm25` が
  `id -u` / `id -g` を渡し、マウント先のディレクトリを先に作る (無いと Docker が root 所有で作る)。
  Jupyter が書き込むホームはコンテナ内の一時領域に置く
- 前処理のノートブックの最後のセルは、組織者が別に配布していた `top1000.train.tsv`
  (リポジトリに無い) を表示するだけで失敗する。生成物には関わらないため、そのセルの失敗は
  許し、生成物 3 つ (文書 JSONL、トピック JSONL、qrels TSV) の有無で成否を決める
- インデックス作成はこの環境で 1 時間 15 分かかる。ノートブックが 1 件ずつ分かち書きして
  Terrier に渡す作りで 1 スレッドしか使わないため、コンテナのリソースを増やしても縮まない。
  一度きりの処理であり、run ファイルが既にあるときは実行しない。作り直すときは `FORCE=1` を渡す
