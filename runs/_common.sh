#!/usr/bin/env bash
# 実行用スクリプトが共通で読み込む設定。

set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

: "${NTCIR_DATA_DIR:=$HOME/dev/dataset/ntcir17}"
export NTCIR_DATA_DIR

# nohup で回すときの出力先。リダイレクト先は起動前に存在している必要がある。
mkdir -p "$REPOSITORY_ROOT/logs"

# 出力をファイルへ流すと数 KB 溜まるまで書き出されず、進捗が見えなくなる。
export PYTHONUNBUFFERED=1

# 使う GPU。共用機のため既定で 1 枚に絞る。
# 例: CUDA_VISIBLE_DEVICES=2,3 ./runs/layer_sweep.sh
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# ADR-0003 の対象モデル。追加するときはここに足す。
MODELS=(
    "cl-tohoku/bert-base-japanese-v3"
    "cl-nagoya/unsup-simcse-ja-base"
)

run_for_each_model() {
    local experiment="$1"
    for model in "${MODELS[@]}"; do
        echo "=============================================================="
        echo " ${experiment} / ${model} / GPU ${CUDA_VISIBLE_DEVICES}"
        echo "=============================================================="
        uv run python run_experiment.py --experiment "$experiment" --model "$model"
        echo
    done
}
