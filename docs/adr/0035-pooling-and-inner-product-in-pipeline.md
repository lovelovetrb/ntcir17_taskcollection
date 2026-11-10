# ADR-0035: プーリングと内積を選べるようにし、キャッシュと記録はモデルの下にプーリングごとに分ける

- Status: Accepted
- Date: 2026-11-10
- ADR-0014 の「検索ベクトルは常に L2 正規化する」、ADR-0026 と ADR-0029 のキャッシュと記録の置き場所を置き換える

## 背景

ADR-0034 で、組織者の DPR_X_TT を CLS と mean の 2 通りで学習し直し、既存の評価のパイプラインで
内積とコサインの両方を測ると決めた。今のパイプラインは次の前提で作られている。

| 場所 | 今の前提 |
|---|---|
| `src/hidden_subspace/encoding/encoder.py` | プーリングは平均 (`mean_pool`) だけ |
| `src/hidden_subspace/vectors.py` の `build_search_vectors` | 最後に必ず L2 正規化する (ADR-0014) |
| `src/hidden_subspace/dapt/models.py` の `resolve_model` | 学習したモデルとして `dapt/` の識別子だけを解決する |
| `sandbox/analysis/pc_removal/run.py` | 対象の 3 モデルを `MODELS` 辞書に書き、1 回の実行で全部回す |
| キャッシュと記録 | `cache/<collection>/<model_id>/`、`results/<collection>/<model_id>/<experiment>/`、`results/pca_analysis/<分析>/<collection>/<model_id>/` で、プーリングの区別がない |

## 決定

### プーリング

- エンコーダでプーリングを `mean` と `cls` から選べるようにする。`cls` は各層の先頭トークンの隠れ状態とする
- キャッシュを読むスクリプト (`run_experiment.py`、`sandbox/analysis/pc_removal`、`src/pca/analysis/*`) は、
  `--model` と `--pooling` でキャッシュを選ぶ
- `--pooling cls` のときは、層 0 (埋め込み層) を評価にも主成分の除去にも使わない (ADR-0034)

### 置き場所

モデルの識別子の下にプーリングを挟む。

```
cache/<collection>/<model_id>/<pooling>/{documents,queries}.pt
results/<collection>/<model_id>/<pooling>/<experiment>/
results/pca_analysis/<分析>/<collection>/<model_id>/<pooling>/
sandbox/analysis/pc_removal/output/<collection>/<model_id>/<pooling>/
```

既存のキャッシュと記録はすべて平均プーリングなので、`mean/` の下に移す。移すのは実装の PR が
approve された時点とし、パスを参照するコードも同じ PR で置き換える。

### 内積

`build_search_vectors` に、最後の L2 正規化をかけるかどうかを呼び出しごとに指定する引数を足す。
かけなければ、検索の点数は正規化しない内積になる。層ごとの正規化 (ADR-0014 の実験軸) はそのまま残す。

### 学習したモデルの解決

- `resolve_model` を `src/hidden_subspace/dapt/models.py` から `src/hidden_subspace/models.py` に移し、
  DAPT と DPR の両方の識別子を解決する
- DPR で学習したモデルは `dpr/<model_id>/<run_name>` という識別子で指し、`cache/dpr/<model_id>/models/<run_name>`
  に置く。DAPT の `dapt/<model_id>/<run_name>` と `cache/dapt/<model_id>/models/<run_name>` と同じ形にする
- DPR の学習のコードは `src/hidden_subspace/dpr/` に置く

### pc_removal のモデルの選び方

`sandbox/analysis/pc_removal/run.py` と `plot.py` の `MODELS` 辞書をやめ、`--model` と `--pooling` の
引数で 1 つずつ回す。記録のファイル名からモデルの短い名前を外し、`metrics.jsonl` と `variance.jsonl` にする。

## 理由

- **プーリングを置き場所に入れる**: 同じモデルの CLS と mean は別のベクトルになる。置き場所に入れないと、
  後から作った方が前のキャッシュを上書きするか、`encode_or_load` が前のキャッシュを黙って読む
- **モデルの識別子の下に挟む**: プーリングはモデルをどう使うかの設定で、同じモデルの中で切り替える。
  モデルの識別子より上に置くと、同じモデルのキャッシュが離れた場所に分かれる
- **正規化を呼び出しごとに指定する**: ADR-0014 は、層ごとの正規化を呼び出しごとに指定する形にしている。
  最後の正規化も同じ形にすれば、既存の呼び出しは「かける」を指定するだけで結果が変わらない
- **`resolve_model` を `dapt/` の外に出す**: DAPT と DPR の両方の識別子を解決するため、どちらかの
  学習のパッケージの中に置くと、もう一方がそのパッケージに依存する
- **pc_removal を引数で選ぶ**: `run_experiment.py` と `src/pca/analysis/*` はすでに `--model` で選んでいる。
  モデルとプーリングの組が増えるため、辞書に足していくより、1 回に 1 つを指定して回す方がそろう

## 結果

- ADR-0014・ADR-0026・ADR-0029 の Status を `Superseded in part by 0035` にする
- 既存のキャッシュ・記録・`sandbox/analysis/pc_removal/output/` の図を `mean/` の下に移す
- `tests/dapt/test_models.py` の `resolve_model` のテストを、移した先に合わせて移す
