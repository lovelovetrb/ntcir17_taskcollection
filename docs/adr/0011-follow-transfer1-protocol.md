# ADR-0011: NTCIR-17 Transfer-1 のプロトコルに準拠する

- Status: Accepted
- Date: 2026-08-30
- ADR-0008 の文書テキスト構成とクエリフィールド、ADR-0010 の適合度の扱いを置き換える

## 背景

NTCIR-17 (2023) に **Transfer タスク**が存在し、その第 1 回 Transfer-1 が
本研究とほぼ同一のデータ設定を採っていることが分かった。

```
train = NTCIR-1   330K+ 文書, 83 トピック
eval  = NTCIR-2   735K  文書, 49 トピック
```

ADR-0002 で独立に決めた「NTCIR-1 で開発 → NTCIR-2 で評価」という分割は、
この公式タスクの設計と同一である。

組織者はこの 2 コレクションを選んだ理由をこう述べている。

> it is known that the relevance judgments of these test collections are
> **much deeper and more thorough than more recent ones**. This enables us to
> evaluate the performance of proposed techniques with a higher level of
> confidence than collections with shallow judgments.

一方で追加判定は行っていない。

> **No additional relevance assessments were performed on submitted runs.**

つまり Transfer-1 参加者は本研究と同一のプールバイアスを受けている（ADR-0007）。
1999 年の NTCIR-1 参加者は自らプールを構成した側であり比較できないが、
**Transfer-1 参加者とは同じ条件で比較できる**。

## 決定

Transfer-1 の第 1 段階検索 (Dense First Stage Retrieval) のプロトコルに準拠する。
組織者の前処理コードから確定した仕様は以下。

### 文書テキスト

```
text = TITL + " " + ABST
```

- `<ABST.P>` タグは除去する
- **`KYWD` は使用しない**
- 区切りは半角スペース 1 個

### クエリ

トピックの **`<TITLE>`** フィールド。

### 適合性判定

qrels は grade 列から数値へ変換する。

```
A → 2,  B → 1,  C → 0
```

評価では **2 値**として扱い、**`rel >= 1` を適合**とする（A と B を適合、C を非適合）。

### 主指標

**nDCG@1000**。

## 理由

- **比較可能性**が得られる。層・次元の手法が確立できた場合に、Transfer-1 の
  参加ランおよび組織者ベースラインと同じ土俵で並べられる
- ADR-0008 / ADR-0010 で選んだ設定に、揃えるのを拒むほどの根拠がなかった。
  DESCRIPTION の採用根拠はマニュアルの「必要な概念をすべて含む」という記載
  だったが、これは TITLE が不十分であることを意味しない。KYWD の追加も
  「効きそう」以上の理由がなかった
- 設定の自由度が減ることで、「その設定だから出た結果ではないか」という批判の
  余地が減る
- 公表値との突き合わせがパイプラインの外部検証になる。ADR-0009 で発生した
  `ntc1-je1` と `ntc1-j1` の取り違えのような事故は、自作のテストでは検出できない
- 分析の粒度が層・次元であることは、測定対象（検索性能）を変えない。設定を
  分ける理由にならない

## 補強された既存の決定

組織者の `ir_datasets` 定義により、以下が公式と一致していることが確認された。

```python
DL_DOCS_TRAIN  = '.../NTCIR-1/mlir/ntc1-j1.utf8.jsonl'               # ADR-0009 と一致
DL_QRELS_TRAIN = '.../NTCIR-1/mlir/rel2_ntc1-j1_0001-0083.utf8.tsv'
DL_DOCS_EVAL   = '.../NTCIR-2/j-docs/ntc12-j1gk.mod.jsonl'           # NTCIR-1 + j1g + j1k
QREL_DEFS_TRAIN = {2: 'relevant', 1: 'partially relevant', 0: 'not relevant'}
```

- 文書集合が `mlir/ntc1-j1` であること（ADR-0009）
- NTCIR-2 の評価には NTCIR-1 の文書を統合する必要があること（`ntc12-j1gk.mod`）
- 利得の数値 `A=2, B=1, C=0` が ADR-0010 の決定と同一であること

なお組織者の前処理コードは `accn[i]` / `titl[i]` / `abst[i]` を添字で対応付けており、
フィールドが 1 つでも欠けるとずれる。`ntc1-j1` は TITL 欠損 0 件・ABST 欠損 0 件
であるため成立するが、`ntc1-je1`（TITL 欠損 3,478 件 / ABST 欠損 5,050 件）では
破綻する。組織者が `ntc1-j1` を選んだ理由がここにも表れている。

## 結果

- ADR-0008 の「文書は TITL→ABST→KYWD」「クエリは DESCRIPTION」は失効する
- ADR-0010 の graded 評価（A=2, B=1 を利得として使う）は失効する。数値の割り当ては
  同じだが、評価時は `rel >= 1` の 2 値として扱う
- ADR-0004 の 512 トークン truncate は影響を受けない。KYWD を落とすことで
  文書はさらに短くなり、truncate の発生率は 1.82% より下がる
- **Phase 1 は NTCIR-1 (train set) を対象とするため、公表値との直接比較はまだ
  できない。** 公表されている参加ランのスコアは eval set (NTCIR-2) のものである。
  Phase 2 に進んだ時点で比較が成立する。今から揃えておくのは、その時点での
  手戻りを避けるためである
- 参考値（KASYS チーム、eval set = NTCIR-2、49 トピック）

  | Run | nDCG@10 | nDCG@20 | nDCG@1000 | MAP |
  |---|---|---|---|---|
  | Contriever 系 最良 | 0.462 | 0.383 | 0.367 | 0.135 |
  | ColBERT-X 単体 | 0.539 | 0.468 | — | 0.229 |
  | ColBERT-X + BM25 (RRF) | 0.562 | 0.502 | — | 0.263 |

## 出典

- [ntcirtransfer/transfer1](https://github.com/ntcirtransfer/transfer1) — 組織者の公式リポジトリ。
  `datasets/ntcir_transfer.py` と `notebooks/preprocess-transfer1-train.ipynb` が仕様の一次情報
- [Overview of the NTCIR-17 Transfer Task](https://repository.nii.ac.jp/records/2001319) (Joho, Keyaki, Oba 2023)
- [KASYS at the NTCIR-17 Transfer Task](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings17/pdf/ntcir/04-NTCIR17-TRANSFER-AbeK.pdf)
- [Building Test Collections for Japanese Dense Information Retrieval Technologies and Beyond](https://ceur-ws.org/Vol-3854/emtcir-4.pdf) (EMTCIR 2024)
