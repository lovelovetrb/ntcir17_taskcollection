# ADR-0032: 分析をまたいで使う道具は src/pca/analysis/common/ に置き、src/pca/core/ は主成分の推定・得点・除去だけにする

- Status: Accepted
- Date: 2026-11-07
- ADR-0029 の「`core/` は分析をまたいで使う部品」を置き換える

## 背景

ADR-0029 は、分析をまたいで使う部品を `src/pca/core/` に置くと決めた。今の `core/` には、
`components.py` だけがある。平均と主成分の推定、主成分得点の計算 (`CommonComponents.scores()`)、
主成分の除去 (`CommonComponents.remove()`) の 3 つで、除去は `sandbox/analysis/pc_removal` が
文書とクエリに使う。

ADR-0030 の文書長の分析では、次の 3 つを `src/pca/analysis/document_length/` の中に書いた。

- `correlation.py`: 主成分得点と値の Pearson の相関係数
- `groups.py`: NTCIR-2 の文書を学会発表と科研費報告書に分ける
- `figures.py`: 主成分得点の散布図

ADR-0031 の助詞・助動詞の割合の分析でも、この 3 つを使う。
この 3 つは主成分の推定とは関係がなく、主成分以外の分析でも使える道具である。

## 決定

```
src/pca/
  core/                 主成分の推定、主成分得点の計算、主成分の除去
  analysis/
    common/             分析をまたいで使う道具 (相関係数、グループ分け、散布図など)
    <分析名>/            分析ごとのスクリプト
```

- `document_length/` の `correlation.py`・`groups.py`・`figures.py` を `analysis/common/` に移す
- `core/` には、主成分そのものを扱う処理 (推定・得点の計算・除去) だけを置く

## 採らなかった案

- **道具も `core/` に置く (ADR-0029 のまま)**: 分析が増えるたびに `core/` に道具が増え、
  主成分の推定と分析の道具の区別がつかなくなる
- **分析ごとに道具を書く**: #89 の分析はどれも同じ相関係数・グループ分け・散布図を使い、
  分析の数だけ同じ処理が重複する

## 結果

- ADR-0029 の Status を `Superseded in part by 0032` にする
