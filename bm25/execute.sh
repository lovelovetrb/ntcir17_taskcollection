#!/usr/bin/env bash
# 組織者のノートブックを順に実行して run ファイルを得る (ADR-0022)。コンテナの中で動く。
#
# FORCE=1 を渡すと、run ファイルがあっても作り直す。

set -euo pipefail

DISTRIBUTION=/data/ntcir/NTCIR-1
COLLECTION=/work/transfer1/testcollections/ntcir/NTCIR-1
NOTEBOOKS=/work/transfer1/notebooks
EXECUTED=/work/transfer1/runs/notebooks
RUN_FILE=/work/transfer1/runs/ntcir17-transfer/train/MyRun-BM25.res.gz
INDEX=/work/transfer1/indexes/ntcir17-transfer/train

# ホストのユーザーで動くため、Jupyter が書き込むホームをコンテナ内の一時領域に置く
export HOME=/tmp/bm25-home
mkdir -p "$HOME"

if [[ -e "$RUN_FILE" && "${FORCE:-0}" != "1" ]]; then
    echo "run ファイルが既にあります: $RUN_FILE"
    echo "作り直すときは FORCE=1 を渡してください。"
    exit 0
fi

# 配布物は読み取り専用でマウントされる。前処理のノートブックは同じ場所に展開するため、写しを作る
mkdir -p "$COLLECTION"
for archive in MLIR.TGZ TOPICS.TGZ; do
    if [[ ! -e "$COLLECTION/$archive" ]]; then
        echo "写す: $DISTRIBUTION/$archive"
        cp "$DISTRIBUTION/$archive" "$COLLECTION/$archive"
    fi
done

mkdir -p "$EXECUTED"
cd "$NOTEBOOKS"

# 前処理の生成物。検索のノートブックはこの 3 つを ir_datasets 経由で読む
PREPROCESSED=(
    "$COLLECTION/mlir/ntc1-j1.utf8.jsonl"
    "$COLLECTION/topics/topic0001-0083.utf8.jsonl"
    "$COLLECTION/mlir/rel2_ntc1-j1_0001-0083.utf8.tsv"
)
preprocessed() {
    for file in "${PREPROCESSED[@]}"; do
        [[ -e "$file" ]] || return 1
    done
}

if ! preprocessed; then
    echo "==== preprocess-transfer1-train.ipynb"
    # 最後のセルは組織者が別に配布していた top1000.train.tsv (リポジトリに無い) を表示するだけで、
    # 生成物には関わらない。そこで止まっても生成物が揃っていれば前処理は済んだとみなす
    papermill preprocess-transfer1-train.ipynb "$EXECUTED/preprocess-transfer1-train.ipynb" --cwd . || true
    if ! preprocessed; then
        echo "前処理の生成物が揃っていません。$EXECUTED/preprocess-transfer1-train.ipynb の出力を確認してください。" >&2
        exit 1
    fi
fi

# ノートブックはインデックスを作り直す前に消さないため、残っていると重ねて書いてしまう
rm -rf "$INDEX"

echo "==== experiment-transfer1-train.ipynb"
papermill experiment-transfer1-train.ipynb "$EXECUTED/experiment-transfer1-train.ipynb" --cwd .

echo
echo "run ファイル: $RUN_FILE"
echo "実行済みノートブック: $EXECUTED"
echo "組織者のノートブックに残る nDCG 0.526288 は、qrels をトピック 0001-0030 だけ読むバグ (2023-06-28 修正) の下で計算された値。"
echo "同じ条件で評価して一致すれば、run は組織者と同一 (ADR-0022):"
python - <<'PY'
import ir_measures
from ir_measures import nDCG
run = list(ir_measures.read_trec_run("/work/transfer1/runs/ntcir17-transfer/train/MyRun-BM25.res.gz"))
base = "/work/transfer1/testcollections/ntcir/NTCIR-1/mlir/"
first30 = list(ir_measures.read_trec_qrels(base + "rel2_ntc1-j1_0001-0030.utf8.tsv"))
all83 = list(ir_measures.read_trec_qrels(base + "rel2_ntc1-j1_0001-0083.utf8.tsv"))
value = ir_measures.calc_aggregate([nDCG], first30, run)[nDCG]
print(f"  トピック 0001-0030 (組織者と同じ条件): nDCG = {value:.6f}  期待値 0.526288  {'一致' if abs(value - 0.526288) < 5e-6 else '不一致'}")
print(f"  トピック 0001-0083 (修正後の qrels):   nDCG = {ir_measures.calc_aggregate([nDCG], all83, run)[nDCG]:.6f}")
print("  いずれも段階的利得 (A=2, B=1) で、層の結果と並べる 2 値の nDCG@1000 ではない")
PY
