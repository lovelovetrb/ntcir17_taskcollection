# ADR-0029: 主成分の分析は src/pca/ に置き、結果は results/pca_analysis/ に分析・コレクション・モデルごとに置く

- Status: **Superseded in part by [ADR-0032](0032-pca-analysis-common.md)、[ADR-0035](0035-pooling-and-inner-product-in-pipeline.md)**
  （ADR-0032 は `core/` に置くものを、ADR-0035 は結果の置き場所を置き換える。以下の本文は決定当時の記録として残す）
- Date: 2026-11-04
- 主成分の分析について、ADR-0024 の「分析は `sandbox/analysis/` に置く」「共通の部品は作らない」
  「図はコミットする」「`results/` は記録専用」を置き換える

## 背景

主成分を除くと検索性能が上がることを `sandbox/analysis/pc_removal` で確かめた (issue #47)。
続けて、各主成分が文書のどんな性質や語と対応するかを、性質ごとに分けて調べる (issue #89)。
主成分の分析がこの研究の主要な部分になったため、sandbox ではなく本体として扱う。

ADR-0024 は分析ごとに処理を書き、共通の部品を作らないと決めていた。#89 の分析は、どれも
同じ手順で主成分を推定し、文書ごとの主成分得点を使う。

## 決定

### コードは src/pca/ に置く

```
src/pca/
  core/                 分析をまたいで使う部品 (平均と主成分の推定など)
  analysis/<分析名>/     分析ごとのスクリプト
```

- `sandbox/analysis/pc_removal` にある平均と主成分の推定を `src/pca/core/` に移す
- `src/hidden_subspace/` の下には置かず、別のパッケージにする

### 結果は results/pca_analysis/ に置く

```
results/pca_analysis/<分析名>/<コレクション>/<model_id>/
  *.jsonl     数値の記録
  figures/    図
```

- `<model_id>` は Hugging Face のモデルの識別子 (`cl-tohoku/bert-base-japanese-v3` など。DAPT のモデルは
  `dapt/<model_id>/<run_name>`) とし、`results/<コレクション>/<model_id>/` と同じ形にする
- 記録も図もコミットしない (`results/` は `.gitignore` の対象)

## 採らなかった案

- **`src/hidden_subspace/pca/` に置く**: `hidden_subspace` は層と次元を選んで検索する実験の
  パッケージで、主成分の分析はそれとは別の関心ごとになる
- **`sandbox/analysis/` に置いたまま、分析ごとに主成分の推定を書く (ADR-0024)**: #89 の分析は
  すべて同じ推定を使うため、分析の数だけ同じ処理が重複する

## 結果

- `pyproject.toml` の `packages` に `src/pca` を足す
- 図は git に残らない。見るにはスクリプトを回して作り直す
