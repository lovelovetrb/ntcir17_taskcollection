# Architecture Decision Records

このディレクトリは、本研究の設計上の決定を時系列で記録する。

## 方針

- 1 決定 = 1 ファイル。`NNNN-短い決定内容.md`
- 決定を**変えるときは既存 ADR を書き換えず**、新しい ADR を追加して古い方の Status を `Superseded by ADR-NNNN` にする
- 「なぜそうしなかったか」を必ず残す。採用理由より却下理由のほうが後から価値が出る
- 未決定のまま進める事項は Status を `Proposed` / `Deferred` にして残す

## Status 一覧

| ADR | 決定 | Status |
|-----|------|--------|
| [0001](0001-j-collection-only.md) | 対象を J コレクションのみとする | Accepted |
| [0002](0002-phase-split-ntcir1-first.md) | Phase 1 は NTCIR-1 単独で行う | Accepted |
| [0003](0003-model-pair.md) | tohoku-bert-v3 と unsup-simcse-ja-base を対象とする | Accepted |
| [0004](0004-truncate-512.md) | 入力長は 512 トークン truncate で統一 | Accepted |
| [0005](0005-not-bound-by-original-protocol.md) | 元タスクの実施手順に忠実である必要はない | Superseded in part by 0011 |
| [0006](0006-phase1-output-definition.md) | Phase 1 の出力は層別性能とランダム次元の分散 | Accepted |
| [0007](0007-evaluation-scope.md) | 全 332,918 件を検索対象とし、未判定は非適合として数える | Accepted |
| [0008](0008-text-construction.md) | 文書は TITL→ABST→KYWD、クエリは DESCRIPTION | Superseded in part by 0011 |
| [0009](0009-document-set-ntc1-j1.md) | 文書集合は `mlir/ntc1-j1` (332,918件) を用いる | Accepted |
| [0010](0010-ndcg-gain-scheme.md) | nDCG の利得は線形の A=2, B=1, C=0 | Superseded in part by 0011 |
| [0011](0011-follow-transfer1-protocol.md) | NTCIR-17 Transfer-1 のプロトコルに準拠する | Accepted |
| [0012](0012-bm25-baseline.md) | BM25 は Transfer-1 の指定構成 (PyTerrier + SudachiPy) で再現 | Accepted |
| [0013](0013-metrics.md) | nDCG / Precision / Recall を k = 1000, 10, 1 で報告 | Accepted |
| [0014](0014-normalization.md) | 層ごとの L2 正規化を次元選択の後に適用し、有無を実験軸とする | Accepted |
| [0015](0015-pooled-cache-only.md) | 全件のプーリング済みキャッシュのみを作る | Accepted |
| [0016](0016-hidden-states-domain-model.md) | 隠れ状態を domain model とし、層と次元の選択を独立させる | Accepted |
| [0017](0017-vector-construction-concatenation.md) | ベクトルの作り方は連結から始める | Accepted |
| [0018](0018-tensor-on-gpu.md) | PyTorch テンソルで保持し、キャッシュ全体を GPU に載せる | Accepted |
| [0019](0019-layer-combinations.md) | 層の組み合わせは 2 層以上の全通りを試す | Accepted |
| [0020](0020-layer-inspector-viewer.md) | 層ごとの検索結果を読むビューアを Streamlit で作る | Superseded in part by 0021 |
| [0021](0021-split-build-and-view.md) | ビューアを結果ファイルの作成と表示に分け、表示側を軽い環境で動かす | Accepted |
