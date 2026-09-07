#!/usr/bin/env bash
# 層ごとの検索性能を、対象モデルすべてで測る (ADR-0006)。
#
# キャッシュがなければ作る。1 モデルあたり 30 分ほどかかるが、2 回目以降は数秒。

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

run_for_each_model "layer-sweep"
