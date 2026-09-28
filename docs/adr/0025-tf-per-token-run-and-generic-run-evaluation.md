# ADR-0025: TF ÷ トークン数 の run を BM25 と同じインデックスから作り、run の評価は置き場所から model_id を導く

- Status: Accepted
- Date: 2026-09-28
- ADR-0023 の「model_id は bm25 に固定」を置き換える

## 決定

### TF ÷ トークン数 の run

浅い層が語の重みを持たない字面一致として振る舞うか (issue #43) を測る物差しとして、
クエリの語ごとに文書中の出現回数を文書のトークン数で割って足した検索を行い、run ファイルに
残す。

```
score(q, d) = Σ_{語 t ∈ q}  tf(t, d) / len(d)
```

語の重み (IDF) も飽和もない。生の出現回数の和にしないのは、長い文書ほど点が高くなる長さの
効果が混ざるため。0 層の平均プーリングは各語の埋め込みを 出現回数 ÷ トークン数 の重みで
足したものなので、この式はそこから導かれる形でもある。

run は ADR-0022 のコンテナで、BM25 と同じ Terrier のインデックス (SudachiPy SplitMode.A の
表層形) に対して作る。したがって `tf(t, d)` と `len(d)` は Sudachi のトークン単位で、BERT の
WordPiece のトークン数ではない。クエリのトークナイズと記号の除去も BM25 のノートブックと
同じにする。重み付けは PyTerrier に Python の関数として渡し、上位 1,000 件を TREC 形式で
書く。スクリプトは `bm25/` に置き、Makefile の `bm25-tf` で動かす。

### 置き場所

```
results/tf-per-token/ntcir17-transfer/train/run.res.gz
```

`results/<model_id>/<experiment>/` の規則に合わせ、model_id を `tf-per-token`、experiment を
BM25 の run と同じ `ntcir17-transfer/train` にする。評価の記録は ADR-0023 と同じく run の
隣に書かれる。

### run の評価は置き場所から model_id と experiment を導く

ADR-0023 の `evaluate_bm25.py` は model_id を `bm25` に固定していた。これを `evaluate_run.py`
に改め、run のパスを `results/` からの相対パスとして読み、先頭のディレクトリを model_id、
run のあるディレクトリまでの残りを experiment とする。

```
results/bm25/ntcir17-transfer/train/MyRun-BM25.res.gz   → model_id bm25,          experiment ntcir17-transfer/train
results/tf-per-token/ntcir17-transfer/train/run.res.gz  → model_id tf-per-token,  experiment ntcir17-transfer/train
```

`results/` の外にある run は拒否する。それ以外 (順位表の作り方、run に無い文書の扱い、記録の
形) は ADR-0023 のまま。

## 検討した選択肢

- **TF の run を `results/bm25/` の下に置き、model_id を bm25 のままにする。** 記録の
  model_id で BM25 と区別できなくなる
- **IDF を外した BM25 (飽和と長さの割引は残す) も作る。** BM25 から IDF の有無だけを
  切り離せるが、問いは「浅い層は字面だけを見ているか」であり、そのためには重みも飽和も
  無い物差しが 1 つあれば足りる。作らない
- **生の出現回数の和。** 長さの効果が混ざり、相関が低くても字面のせいか長さのせいか
  分けられない
- **BERT の WordPiece でトークナイズして TF を数える。** BM25 と別のトークナイズになり、
  BM25 との比較に別の差が入る。Sudachi のトークン数で割る近似を取る

## 結果

- `evaluate_bm25.py` は `evaluate_run.py` に名前が変わり、`--run` の指定で BM25 と TF の
  どちらも評価できる。Makefile の `bm25-evaluate` はそのまま BM25 の run を評価する
- TF の run と記録は `results/tf-per-token/` に置かれ、`.gitignore` の対象
- 物差しは BM25 と TF ÷ トークン数 の 2 つになり、#43 はこの 2 つと各層の相関を並べる
