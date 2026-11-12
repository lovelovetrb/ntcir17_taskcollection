# 分析

層の機序を確かめる定量分析 (ADR-0024)。分析ごとにディレクトリを切り、スクリプトを置く。図と表はその下の `output/` に書き出し、コミットする。
共通の部品は作らない。

| ディレクトリ | issue | 内容 |
|---|---|---|
| `lexical_correlation/` | #43 | 層ごとのトピック別 nDCG@1000 と BM25 / TF ÷ トークン数 との相関 |
| `word_sense/` | #46 | 浅い側・深い側の上位文書の語義を Gemini に判定させ、別の意味で解釈した割合を層で比べる |
| `pc_removal/` | #47 | 層ごとに文書集合の上位 k 本の主成分を除いたときの nDCG@1000 の変化 (全トピック平均と、生基準・k=0 基準のトピック別) と、除いた主成分の分散占有率 |
| `pos_frequency/` | #91 | NTCIR-2 の品詞の大分類ごとの出現回数と、その品詞の語を含む文書の数 |

実行はリポジトリのルートから。記録 (`results/ntcir1/`) が揃っている必要がある。分析は NTCIR-1 だけを対象にする。

```
PYTHONPATH=sandbox uv run python -m analysis.lexical_correlation
```

`word_sense/` は Gemini の API キーを環境変数 `GEMINI_API_KEY` で受け取る (`.env.example` 参照)。
判定の生データは `sandbox/results/word_sense/` (git 管理外) に置く。

```
CUDA_VISIBLE_DEVICES=<DEVICE_ID> PYTHONPATH=sandbox uv run --env-file .env python -m analysis.word_sense.collect_judgements --model bert
```

`pc_removal/` は記録ではなくキャッシュ (`cache/<collection>/<model_id>/<pooling>/`) を読み、実験 (GPU を使う) と描画を分けて回す。NTCIR-1 と NTCIR-2 を `--collection` で選び、モデルとプーリングは `--model` と `--pooling` で 1 つずつ選び、記録と図は `output/<collection>/<model_id>/<pooling>/` に置く。`--collection ntcir2-unique` では NTCIR-2 にだけある文書から平均と主成分を推定し、`output/ntcir2-unique/<model_id>/<pooling>/` に置く。記録 (`summary.jsonl`・`topic/<topic_id>.jsonl`・`variance.jsonl`) は本体の実験と同じ形で、コミットしない。指標の変化の図は類似度 (`cosine` / `inner_product`) ごとに描く。

```
CUDA_VISIBLE_DEVICES=<番号> PYTHONPATH=sandbox uv run python -m analysis.pc_removal.run --collection ntcir2 \
    --model cl-tohoku/bert-base-japanese-v3 --pooling mean
PYTHONPATH=sandbox uv run python -m analysis.pc_removal.plot --collection ntcir2 \
    --model cl-tohoku/bert-base-japanese-v3 --pooling mean
CUDA_VISIBLE_DEVICES=<番号> PYTHONPATH=sandbox uv run python -m analysis.pc_removal.run --collection ntcir2-unique \
    --model cl-tohoku/bert-base-japanese-v3 --pooling mean
PYTHONPATH=sandbox uv run python -m analysis.pc_removal.plot --collection ntcir2-unique \
    --model cl-tohoku/bert-base-japanese-v3 --pooling mean
```

`pos_frequency/` は NTCIR-2 の配布物を読み、`src/pca/analysis/common/words.py` と同じく 512 トークンで打ち切った後の文章を MeCab で分けて数える。表は `output/ntcir2.md` に書く。

```
PYTHONPATH=sandbox uv run python -m analysis.pos_frequency --model cl-tohoku/bert-base-japanese-v3
```
