# ADR-0036: DPR の学習データは mMARCO の三つ組から作り、Tevatron の既定値で GPU 1 枚で学習する

- Status: Accepted
- Date: 2026-11-13

## 背景

ADR-0034 で、組織者の DPR_X_TT (xlm-roberta-large を mMARCO の日本語版で、Tevatron で学習) を CLS と mean の
2 通りで学習し直すと決めた。ADR-0035 で、学習のコードは `src/dpr/` に置き、学習したモデルは
`dpr/<model_id>/<run_name>` という識別子で `cache/dpr/<model_id>/models/<run_name>/` に置くと決めた。
学習データの作り方、Tevatron の入れ方、計算資源、コードの形は決めていない。

mMARCO (`unicamp-dl/mmarco`) は、MS MARCO を機械翻訳したデータである。日本語版の学習に使うのは次の 3 つの
タブ区切りのファイルで、どれも Google 翻訳の版 (`data/google/`) と、言語によらない三つ組である。

| ファイル | 1 行の形 | 大きさ |
|---|---|---|
| `data/triples.train.ids.small.tsv` | クエリ ID・適合文書の ID・適合しない文書の ID | 905 MB、39,780,811 行 |
| `data/google/queries/train/japanese_queries.train.tsv` | クエリ ID・クエリ文 | 46 MB、808,731 行 |
| `data/google/collections/japanese_collection.tsv` | 文書 ID・本文 | 3.9 GB |

三つ組をクエリごとにまとめ、適合しない文書を重複なしで数えた。

| 項目 | 値 |
|---|---|
| クエリ | 398,792 |
| 適合しない文書の数 (最小 / 中央値 / 平均 / 最大) | 1 / 99 / 99.3 / 524 |
| 適合しない文書が 7 件未満のクエリ | 7,550 (1.9%) |
| 適合文書が 2 件以上のクエリ | 15,247 (最大 7 件) |

Tevatron (main) は、1 クエリを 1 行にした jsonl (`query_id`・`query`・`positive_passages`・
`negative_passages`) を受け付ける。学習のたびに、適合文書から 1 件、負例から `train_group_size` − 1 件を選ぶ。
負例が足りなければ、重複を許して選ぶ (`random.choices`)。

## 決定

### 学習データ

mMARCO の三つ組をクエリごとにまとめ、ID を日本語のクエリ文と本文に置き換えて、Tevatron の jsonl にする。

- 負例は、三つ組でそのクエリと組になっている適合しない文書すべてとする
- 適合文書が複数あるクエリは、すべてを `positive_passages` に入れる
- 負例が 7 件未満のクエリも除かない

```
cache/dpr/data/
  mmarco/       取ってきた 3 つのファイル (そのまま)
  train.jsonl   Tevatron の形にした学習データ
```

### データの取り方

`huggingface_hub` で 3 つのファイルを直接取る。取るスクリプトは `src/dpr/download.py` に置き、
`python -m dpr.download` で回す。

### Tevatron

最新の main を `pyproject.toml` に git の依存として入れ、コミットは `uv.lock` で固定する。

### 学習の設定

GPU 1 枚で、Tevatron の既定値で学習する。ADR-0034 で決めた値と、既定値を設定ファイルに書く。

| 項目 | 値 |
|---|---|
| モデル | `FacebookAI/xlm-roberta-large` |
| プーリング | `cls` / `mean` (設定ファイルごと) |
| 文書の最大長 / クエリの最大長 | 512 / 32 |
| `train_group_size` | 8 |
| GPU 1 枚あたりのバッチ | 8 |
| エポック | 3 |
| 学習率 | 5e-5 |
| warmup | 10% |
| 乱数のシード | 42 |

### コードの形

DAPT (`src/hidden_subspace/dapt/`) と同じ形にする。

```
src/dpr/
  README.md      概要、データの取り方、学習の回し方、学習したモデルの置き場所と識別子
  __main__.py    学習の入口 (python -m dpr --config src/dpr/config/cls.yaml)
  download.py    mMARCO の 3 つのファイルを取る
  config/
    cls.yaml
    mean.yaml
```

- 使う GPU は `CUDA_VISIBLE_DEVICES` で選ぶ
- `run_name` は設定の値から作り、どれか 1 つを変えれば別の名前になるようにする。プーリングも入れる
- 学習の終わりに、モデルとトークナイザを `cache/dpr/<model_id>/models/<run_name>/` に保存する。
  `resolve_model` が確かめる `config.json`・`model.safetensors`・`tokenizer_config.json` がそろうようにする
- 学習の記録は DAPT と同じ wandb の project に送る

## 理由

- **mMARCO の三つ組から作る**: クエリ・文書・組み合わせが、すべて mMARCO の中で完結する。概要論文の
  「mMARCO の日本語版で学習した」とも一致する
- **負例が少ないクエリも除かない**: Tevatron の既定の動き (重複を許して選ぶ) に任せる。重複した負例が
  入るのは全クエリの 1.9% である
- **ファイルを直接取る**: mMARCO を `datasets` で読むには読み込みスクリプト (`mmarco.py`) が要るが、
  本プロジェクトの `datasets>=5.1.0` ではこの形式を使えない
- **最新の main を使う**: プーリングを選ぶ引数 (`--pooling`) は main にしかなく、0.1.0 では mean で学習できない
- **GPU 1 枚**: バッチ内の他のクエリの文書も負例になるため、GPU の枚数で負例の数が変わる。1 枚に固定し、
  既定値どおりの数にする
- **設定ファイルを `config/` に分ける**: CLS と mean の 2 つがあり、並べて置く

## 結果

- `pyproject.toml` に Tevatron と `huggingface_hub` を足す。transformers 5 系で動くかは、依存を足すときに確かめる
- `pyproject.toml` の `packages` に `src/dpr` を足す (ADR-0035)
- 1 エポックは 49,849 ステップ、3 エポックで 149,547 ステップになる。1 ステップにかかる時間は、
  最初に小さく回すときに測る
