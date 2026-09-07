#!/usr/bin/env bash
# 2 層以上のすべての組み合わせで検索性能を測る (ADR-0019)。
#
# 層ごとの正規化の有無で実験を分けて回す。1 実験あたり 10〜12 分、4 回で約 45 分。
# 13 層の構成はピークで 37.3 GB を使うため、80 GB のカードを指定すること。

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_for_each_model "layer-combinations"
run_for_each_model "layer-combinations-normalized"
