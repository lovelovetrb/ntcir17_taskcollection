# DPR の学習

組織者の DPR_X_TT (xlm-roberta-large を mMARCO の日本語版で Tevatron で学習) を、CLS と mean の 2 通りで学び直す (ADR-0034・ADR-0036)。

## データの取り方

mMARCO (`unicamp-dl/mmarco`) から、次の 3 つのファイルを `huggingface_hub` で取る。置き場所は `cache/dpr/data/mmarco/` で、その下は mMARCO のリポジトリと同じ並びになる。

| ファイル | 中身 |
|---|---|
| `data/triples.train.ids.small.tsv` | クエリ ID・適合文書の ID・適合しない文書の ID |
| `data/google/queries/train/japanese_queries.train.tsv` | クエリ ID・クエリ文 |
| `data/google/collections/japanese_collection.tsv` | 文書 ID・本文 |

```
uv run python -m dpr.download
```

学習データ `cache/dpr/data/train.jsonl` は、学習を始めるときになければ作る。三つ組をクエリごとにまとめ、ID を日本語のクエリ文と本文に置き換えた、1 クエリ 1 行の Tevatron の形 (`positive_passages`・`negative_passages`) にする。三つ組の ID が日本語のクエリと文書集合にない場合は止まる。

## 学習の回し方

GPU は `CUDA_VISIBLE_DEVICES` で 1 枚だけ選ぶ。バッチ内の他のクエリの文書も負例になるため、GPU の枚数で負例の数が変わる。

```
CUDA_VISIBLE_DEVICES=<番号> uv run python -m dpr --config src/dpr/config/cls.yaml
CUDA_VISIBLE_DEVICES=<番号> uv run python -m dpr --config src/dpr/config/mean.yaml
```

設定ファイル (`config/cls.yaml`・`config/mean.yaml`) の違いはプーリングだけである。設定ファイルにない値は Tevatron の既定値になる。学習の記録は wandb に送る。

## 学習したモデルの置き場所と識別子

学習したモデルとトークナイザは `cache/dpr/<model_id>/models/<run_name>/` に置く。`run_name` は学習の結果を変える設定の値から作る。

評価の側からは `dpr/<model_id>/<run_name>` という識別子で指す (ADR-0035)。

```
python run_experiment.py --collection ntcir2 --experiment layer-sweep \
    --model dpr/FacebookAI/xlm-roberta-large/<run_name> --pooling cls
```
