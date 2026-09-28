# ADR-0023: BM25 の run を層と同じ定義で評価し、同じ形の記録として run の隣に置く

- Status: Accepted
- Date: 2026-09-28
- ADR-0022 の「run ファイルを層の結果と同じ定義で評価する処理」を決める

## 決定

ADR-0022 で得た run ファイル (`results/bm25/ntcir17-transfer/train/MyRun-BM25.res.gz`) を、
層の実験と同じ `align_relevance` と `evaluate` に通し、同じ `CUTOFFS` (1, 10, 1000) で指標を
計算する。適合は ADR-0011 の 2 値 (A と B) であり、run に対して別の評価器は使わない。

### run から順位表を作る

`evaluate` は全文書を並べた `Ranking` を受け取るが、run には各トピック上位 1,000 件しかない。
そこで、run にある文書を run の順位どおりに先頭に置き、残りの文書をコレクションの並び順で
後ろに埋めて `Ranking` にする。

```
順位 1 .. n      run に書かれた文書 (n ≤ 1000)。スコアは run の値
順位 n+1 ..      run に無い文書。コレクションの並び順。スコアは run のどの値より小さい値
```

k ≤ 1000 の指標は埋めた部分に影響されない。`CUTOFFS` の最大は 1000 なので、埋め方が指標に
現れることはない。1,000 件に満たないトピック (クエリの語を含む文書が少ないもの。今回の
run では 8 トピック) も同じ規則で埋める。run にあってコレクションに無い文書 ID は、
別のコレクションの run を渡した印なので拒否する。

この変換は `hidden_subspace/trec_run.py` に置く。run ファイルを読む関数と、`Ranking` に
する関数の 2 つで、隠れ状態にも BM25 にも依存しない。

### 記録

記録は既存の型 (`ConfigurationResult`、`TopicResult`) と `write_results` をそのまま使い、
`results_of_evaluation` (ADR なし、#37) で作る。BM25 のための型は作らない。

```
model_id          "bm25"
experiment        "ntcir17-transfer/train"     run のあるディレクトリ (ノートブックの $RUN)
layers            []
dimensions        {"kind": "all"}
normalize_layers  false
zero_norm         queries 0 / documents 0
```

`layers` 以下の 4 つは隠れ状態の検索にしか意味がなく、BM25 の記録では値を持たない印として
置く。読む側は `model_id` が `bm25` のときこれらを解釈しない。

書き出し先は `write_results` の規則どおり `results/<model_id>/<experiment>/` になり、
run の隣に並ぶ。

```
results/bm25/ntcir17-transfer/train/MyRun-BM25.res.gz   run (ADR-0022 のコンテナが置く)
results/bm25/ntcir17-transfer/train/summary.jsonl       全トピック平均
results/bm25/ntcir17-transfer/train/topic/0001.jsonl    トピックごと
```

読むときは `read_topic_results(root, "bm25", "ntcir17-transfer/train")` で、層の記録と同じ
関数で読める。

### エントリポイント

`run_experiment.py` と同じ階層に `evaluate_bm25.py` を置く。本体の環境 (uv) で動き、
コンテナも GPU も要らない。流れは `run_experiment.py` から検索を抜いたもので、
コレクションを読む、run を読んで順位表にする、評価する、記録にする、書き出す、表を出す、
の順。Makefile には `bm25-evaluate` を足す。

## 検討した選択肢

- **BM25 用の記録の型を作る。** `Configuration` の代わりに run の場所を持つ型と、その
  書き出し・読み込みを別に書く案。層の記録に無意味な値が入らない利点はあるが、この 1 件の
  ために型と入出力を二重に持つのは過剰と判断した
- **`TopicResult` に run のフィールドを足し、`Configuration` を省略可能にする。** 層の記録
  すべてに使わないフィールドが増え、`from_record` に分岐が要る
- **experiment に run ファイルのパスを入れる。** `write_results` が同じパスにディレクトリを
  作ろうとして、既にある run ファイルと衝突する。ディレクトリまでにすれば衝突せず、run と
  記録が同じ場所に並ぶ
- **`run_experiment` に BM25 の構成を足す。** `run_experiment` は隠れ状態と
  `ConfigurationPlan` を前提にしており、BM25 を押し込むと両方が歪む
- **コンテナ内で ir_measures を使って評価する。** 段階的利得の nDCG になり (ADR-0022)、
  層と定義が揃わない。環境も分かれる

## 結果

- 層の結果と BM25 が、同じ qrels、同じ 2 値の定義、同じ関数で計算した指標で並ぶ
- 記録の `layers` などに意味のない値が入る。読む側がそれを解釈しないことが前提になる
- run に無いトピックがあれば、そのトピックはコレクションの並び順だけの順位表になり、指標は
  ほぼ 0 になる。今回の run は 83 トピックすべてを含む
