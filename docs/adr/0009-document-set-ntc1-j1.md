# ADR-0009: 文書集合は mlir/ntc1-j1 を用いる

- Status: Accepted
- Date: 2026-08-30

## 背景

NTCIR-1 には日本語文書を含む集合が 2 つ存在し、当初これを取り違えていた。

| tarball | 文書集合 | 件数 | 内容 |
|---|---|---|---|
| `ADHOC.TGZ` | `adhoc/ntc1-je1` | 339,483 | 日英両方のフィールドを持つ (JE Collection) |
| `MLIR.TGZ` | `mlir/ntc1-j1` | **332,918** | 日本語フィールドのみ (J Collection) |
| `CLIR.TGZ` | `clir/ntc1-e1` | — | 英語フィールドのみ (E Collection) |

それぞれに専用の適合性判定ファイルが付属する。`MANUAL-E.PDF` Fig. 5-2 に
公式の組合せが定められている。

```
TASK           DOCUMENTS       TOPICS                RELEVANCE JUDGMENTS
ad hoc         adhoc/ntc1-je1  topic0001-0030 (30)   adhoc/rel*_ntc1-je1_0001-0030
ad hoc         adhoc/ntc1-je1  topic0031-0083 (53)   adhoc/rel*_ntc1-je1_0031-0083
monolingual    mlir/ntc1-j1    topic0001-0030 (30)   mlir/rel*_ntc1-j1_0001-0030
monolingual    mlir/ntc1-j1    topic0031-0083 (53)   mlir/rel*_ntc1-j1_0031-0083
cross-lingual  clir/ntc1-e1    topic0001-0030 (21)   clir/rel*_ntc1-e1_0001-0030
cross-lingual  clir/ntc1-e1    topic0031-0083 (39)   clir/rel*_ntc1-e1_0031-0083
```

## 決定

文書集合は **`mlir/ntc1-j1`（332,918 件）** を用いる。

## 理由

リポジトリに置かれている `ntcir1/rel1_j1_all` は、
`mlir/rel1_ntc1-j1_0001-0030` と `mlir/rel1_ntc1-j1_0031-0083` を連結したものである
ことを md5 で確認した。

```
b4fe342af2b8860d2d94ee260db8668d  ntcir1/rel1_j1_all
b4fe342af2b8860d2d94ee260db8668d  cat mlir/rel1_ntc1-j1_0001-0030 mlir/rel1_ntc1-j1_0031-0083
```

したがって Fig. 5-2 の組合せに従い、対応する文書集合は `mlir/ntc1-j1` となる。
これは ADR-0001（J コレクションのみ）とも整合する。

加えて `ntc1-j1` はデータ品質でも優れている。

| | `ntc1-j1` | `ntc1-je1` |
|---|---|---|
| 件数 | 332,918 | 339,483 |
| `TITL` 欠損 | **0** | 3,478 |
| `ABST` 欠損 | **0** | 5,050 |
| `KYWD` 欠損 | 19,205 | 21,227 |
| 3 フィールド全欠損 | **0** | 1,419 |

`ntc1-j1` は全件が日本語タイトルと抄録を持ち、空文書が存在しない。

## 結果

- 適合性判定のカバレッジ（実測）

  ```
  判定済み文書  120,359 件  → コーパス外  2 件
  A 判定文書      4,853 件  → コーパス外  1 件
  A|B 判定文書    5,540 件  → コーパス外  1 件
  判定済みが全文書に占める割合  120,359 / 332,918 = 36.2%
  ```

  完全な一致ではない。qrels に現れるがコーパスに存在しない ID が 2 件あり、
  うち 1 件は A 判定である。件数として無視できるが、**パーサはこれで落ちては
  ならない**。回帰テストのケースとして残す
- ADR-0002 / ADR-0004 / ADR-0008 に記載していた測定値は `ntc1-je1` に対する
  ものだった。`ntc1-j1` の値に修正済み
- 文書 ID は `gakkai-XXXXXXXXXX` 形式で、qrels と同一。NTCIR-1 内では ID 変換は
  不要である（`gakkai-` → `gakkai-j-` の変換が要るのは Phase 2 で NTCIR-2 の
  qrels と突き合わせるときのみ）
