# 分析

層の機序を確かめる定量分析 (ADR-0024)。分析ごとにディレクトリを切り、スクリプトと図を同じ階層に置く。
共通の部品は作らない。

| ディレクトリ | issue | 内容 |
|---|---|---|
| `lexical_correlation/` | #43 | 層ごとのトピック別 nDCG@1000 と BM25 / TF ÷ トークン数 との相関 |

実行はリポジトリのルートから。記録 (`results/`) が揃っている必要がある。

```
PYTHONPATH=sandbox uv run python -m analysis.lexical_correlation
```
