# 分析

層の機序を確かめる定量分析 (ADR-0024)。分析ごとにディレクトリを切り、スクリプトを置く。図と表はその下の `output/` に書き出し、コミットする。
共通の部品は作らない。

| ディレクトリ | issue | 内容 |
|---|---|---|
| `lexical_correlation/` | #43 | 層ごとのトピック別 nDCG@1000 と BM25 / TF ÷ トークン数 との相関 |
| `word_sense/` | #46 | 浅い側・深い側の上位文書の語義を Gemini に判定させ、別の意味で解釈した割合を層で比べる |

実行はリポジトリのルートから。記録 (`results/`) が揃っている必要がある。

```
PYTHONPATH=sandbox uv run python -m analysis.lexical_correlation
```

`word_sense/` は Gemini の API キーを環境変数 `GEMINI_API_KEY` で受け取る (`.env.example` 参照)。
判定の生データは `sandbox/results/word_sense/` (git 管理外) に置く。

```
PYTHONPATH=sandbox uv run python -m analysis.word_sense.collect_judgements --model bert
```
