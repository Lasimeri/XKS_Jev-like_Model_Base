#!/usr/bin/env bash
# tune.sh [OUT_DIR]: one bench round per configuration on
# tests/bench/tune.jsonl (12 developer-task requests), each configuration in
# its own process, the baseline first and last (a drift control). Prints the
# cold, warm and single decisions/s per configuration; full tables in OUT_DIR.
set -euo pipefail
out=${1:-out/tune}
mkdir -p "$out"
bin=./target/release/jevbase
file=tests/bench/tune.jsonl

run() {
    local name=$1
    shift
    env "$@" bench --rounds 1 "$file" >"$out/$name.txt" 2>&1
    printf '%-10s' "$name"
    awk '/decisions\/s min/ { for (i = 1; i <= NF; i++) if ($i == "median") m = $(i + 1); mode = $0; sub(/^ *[^ ]+ /, "", mode); sub(/: decisions.*/, "", mode); printf "  %s %s", mode, m } END { print "" }' "$out/$name.txt"
}

run base-a  "$bin" -t 8
run t6      "$bin" -t 6
run t4      "$bin" -t 4
run pin8    OMP_PLACES=cores OMP_PROC_BIND=close "$bin" -t 8
run fa-off  "$bin" -t 8 --set flash_attn=off
run fa-on   "$bin" -t 8 --set flash_attn=on
run base-b  "$bin" -t 8
