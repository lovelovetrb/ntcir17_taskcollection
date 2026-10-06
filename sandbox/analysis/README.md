# 分析

層の機序を確かめる定量分析 (ADR-0024)。分析ごとにディレクトリを切り、スクリプトを置く。図と表はその下の `output/` に書き出し、コミットする。
共通の部品は作らない。

| ディレクトリ | issue | 内容 |
|---|---|---|
| `lexical_correlation/` | #43 | 層ごとのトピック別 nDCG@1000 と BM25 / TF ÷ トークン数 との相関 |
| `pc_removal/` | #47 | 層ごとに文書集合の上位 k 本の主成分を除いたときの nDCG@1000 の変化 (全トピック平均とトピック別) |

実行はリポジトリのルートから。記録 (`results/`) が揃っている必要がある。

```
PYTHONPATH=sandbox uv run python -m analysis.lexical_correlation
```

`pc_removal/` は記録ではなくキャッシュ (`cache/`) を読み、実験 (GPU を使う) と描画を分けて回す。記録 `output/metrics-<model>.jsonl` はコミットしない。

```
CUDA_VISIBLE_DEVICES=<番号> PYTHONPATH=sandbox uv run python -m analysis.pc_removal.run
PYTHONPATH=sandbox uv run python -m analysis.pc_removal.plot
```
