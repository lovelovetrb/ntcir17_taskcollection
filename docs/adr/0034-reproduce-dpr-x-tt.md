# ADR-0034: 組織者の DPR_X_TT を CLS と mean の 2 通りで学習し直し、主成分の除去を全層に当てる

- Status: **Superseded in part by [ADR-0035](0035-pooling-and-inner-product-in-pipeline.md)**
  （再現できなかったときに 128 トークンで測り直す手順を取り消す。以下の本文は決定当時の記録として残す）
- Date: 2026-11-10

## 背景

主成分を除くと nDCG@1000 が上がることを、`cl-tohoku/bert-base-japanese-v3` と
`cl-nagoya/unsup-simcse-ja-base` の両方で確かめた (`sandbox/analysis/pc_removal`、issue #47)。
ただし、どちらも検索向けの学習をしていないモデルで、除く前の値が低い。NTCIR-2 で除く前に最も高いのは
SimCSE の層 12 の 0.164、除いた後に最も高いのは層 0 の 0.236 である。

NTCIR-17 Transfer の概要論文 (Joho ら 2023) によると、組織者は Dense First Stage Retrieval に
6 つのベースラインを出している。値は概要論文の Figure 1 (NTCIR-2、nDCG@1000) から読んだ。

| ラン | 方式 | 元のモデル | 学習データ | nDCG@1000 |
|---|---|---|---|---|
| ColBERT_X_ZS | ColBERT v1 | xlm-roberta-large | MS MARCO (英語) | 0.446 |
| ColBERT_X_TT | ColBERT v1 | xlm-roberta-large | mMARCO の日本語版 | 0.406 |
| DPR_X_TT | DPR (Tevatron) | xlm-roberta-large | mMARCO の日本語版 | 0.353 |
| DPR_X_ZS | DPR (Tevatron) | xlm-roberta-large | MS MARCO (英語) | 0.339 |
| ColBERT_J | ColBERT v1 | bert-large-japanese-v2 | mMARCO の日本語版 | 0.330 |
| DPR_J | DPR (Tevatron) | bert-large-japanese-v2 | mMARCO の日本語版 | 0.301 |

概要論文に書かれているのは、元のモデル・学習データ・枠組みだけである。プーリング、類似度、
クエリと文書で重みを共有するか、最大長、エポック数は書かれていない。組織者のリポジトリ
(`ntcirtransfer/transfer1`) にも、これらのランのコードとチェックポイントはない。

## 決定

### 再現するラン

組織者の **DPR_X_TT** を学習し直す。xlm-roberta-large を mMARCO の日本語版で、Tevatron を使って学習する。

### 学習の設定

- Tevatron の既定値を使う。クエリと文書で重みを共有し (`untie_encoder` を使わない)、
  ベクトルを正規化せず、点数は内積とする
- プーリングは **CLS と mean の 2 通り**で学習する (`--pooling cls` / `--pooling mean`)
- 文書の最大長 (`passage_max_len`) は 512 にする。クエリの最大長は既定値の 32 のままとする

### 再現できたとみなす条件

CLS で学習したモデルを、内積で、主成分を除かずに NTCIR-2 で評価した nDCG@1000 が **0.353 ± 0.02**
に入ること。

入らない場合は、同じモデルで文書を Tevatron の既定値の 128 トークンで打ち切ってベクトルにし直し、
測り直す。組織者の論文に最大長が書かれていないため。

### 評価

- 既存の評価のパイプラインを使い、新しく作るのは学習のパイプラインだけとする
- 主成分の除去は、`sandbox/analysis/pc_removal` と同じ k で**全層**に当てる。mean は埋め込み層と 24 層、
  CLS は 24 層 (埋め込み層を除く)。CLS の埋め込み層は、評価もしない
- 類似度は **内積とコサインの両方**を記録する。主な比較は内積で行う。コサインは、
  `cl-tohoku/bert-base-japanese-v3` と `cl-nagoya/unsup-simcse-ja-base` の結果と同じ条件で比べるために使う
- 主指標は nDCG@1000 とする。ADR-0013 の他の指標も、これまでと同じく記録する
- CLS と mean は別々に回す

## 理由

- **DPR を選ぶ**: DPR は文書を 1 本のベクトルにする。手元の主成分の除去は文書ベクトル 1 本に対する
  操作なので、同じ操作をそのまま当てられる。ColBERT はトークンごとのベクトルで照合するため、
  同じ操作にならない
- **TT を選ぶ**: 学習データも NTCIR-2 も日本語である。DPR の 2 つのうち、nDCG@1000 も高い
- **CLS と mean の両方**: Tevatron の既定値は CLS である。一方、手元で見つけた主成分と文書の性質の対応
  (ADR-0030・ADR-0033) は、平均プーリングで得たものである。CLS で組織者の値と比べ、mean で手元の結果と
  同じプーリングで比べる
- **CLS の埋め込み層を外す**: 埋め込み層の出力は Attention を通らず、先頭トークンの埋め込みと位置 0 の
  位置埋め込みだけで決まるため、CLS のベクトルは全文書で同じになる。`cl-tohoku/bert-base-japanese-v3` に
  内容の違う 2 文を通すと、CLS の差の最大値は層 0 で 0.0、層 1 で 0.213、層 12 で 1.016 だった。
  全文書が同じベクトルでは分散が 0 になり、主成分を推定できず、検索しても全文書が同点になる
- **内積を主にする**: DPR_X_TT は正規化しない内積で学習する。コサインに変えると、変えただけで値が動き、
  主成分を除いた効果と混ざる
- **最大長を 512 にする**: NTCIR-2 の文書から無作為に選んだ 1 万件を xlm-roberta-large のトークナイザで
  数えると、中央値は 241 トークンで、92.2% が 128 トークンを超え、14.5% が 512 トークンを超えた。
  Tevatron の既定値の 128 では、9 割以上の文書が冒頭しか読まれない。512 は xlm-roberta-large が
  受け付ける最大長で (`max_position_embeddings` = 514 のうち 2 つは位置の番号のずれに使う)、
  ADR-0004 の打ち切り長とも同じである。クエリは最長 17 トークンで、32 に収まる

## 採らなかった案

- **ColBERT_X_ZS (0.446) を再現する**: 組織者のランで最も高いが、照合がトークンごとで、文書ベクトル
  1 本に対する主成分の除去と同じ操作にならない
- **DPR_X_ZS を再現する**: 学習データが英語で、NTCIR-2 との間に言語の違いも入る。nDCG@1000 も TT より低い
- **CLS だけで学習する**: 機序が変わったときに、学習によるものかプーリングによるものかを分けられない
- **mean だけで学習する**: 組織者の値と比べて、再現できたかを確かめられない

## 結果

既存のパイプラインに要る変更は、ADR-0035 で決める。

キャッシュは、NTCIR-2 の全文書 (736,158 件) を 25 層 × 1,024 次元の fp16 で持つと 35.1 GiB になる。
CLS と mean を別々に回すため、GPU に載るのは 1 本ずつで、ADR-0018 のまま 80 GB の GPU に収まる。

## 出典

- [Overview of the NTCIR-17 Transfer Task](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings17/pdf/ntcir/01-NTCIR17-OV-TRANSFER-JohoH.pdf) (Joho, Keyaki, Oba 2023)
- [texttron/tevatron](https://github.com/texttron/tevatron) — `src/tevatron/retriever/arguments.py` と `src/tevatron/retriever/modeling/dense.py` の既定値
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) (Karpukhin ら 2020)
