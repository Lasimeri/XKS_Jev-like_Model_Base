#!/usr/bin/env bash
# serve-and-ask.sh PORT REQUESTS OUT -- SERVER COMMAND...
#
# Starts SERVER COMMAND (a POST /v1/systemone server that listens on PORT),
# waits for GET /health, posts every request line of REQUESTS in order (blank
# lines and lines starting with # are skipped), writes one response per line
# to OUT, prints the server's resident memory, then stops it. The server's
# log goes beside OUT (.log). Requests go in order on one fresh server, so
# the session cache is reused the same way by any engine that keeps one.
set -euo pipefail

if [[ $# -lt 5 || $4 != -- ]]; then
    echo "usage: $0 PORT REQUESTS OUT -- SERVER COMMAND..." >&2
    exit 2
fi
port=$1 requests=$2 out=$3
shift 4
log=${out%.jsonl}.log
url=http://127.0.0.1:$port

"$@" >"$log" 2>&1 &
pid=$!
trap 'kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true' EXIT

up=0
for _ in $(seq 1200); do
    if curl -sf "$url/health" >/dev/null; then up=1; break; fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 0.25
done
if [[ $up -ne 1 ]]; then
    echo "the server did not answer $url/health; see $log" >&2
    exit 1
fi

: >"$out"
n=0
t0=$(date +%s%N)
while IFS= read -r line; do
    [[ -z $line || $line == \#* ]] && continue
    n=$((n + 1))
    curl -s -H 'content-type: application/json' --data-binary "$line" \
        -w '\n' "$url/v1/systemone" >>"$out"
done <"$requests"
t1=$(date +%s%N)

rss_kb=$(ps -o rss= -p "$pid" | tr -d ' ')
errors=$(grep -c '"error"' "$out" || true)
awk -v n="$n" -v e="$errors" -v ns="$((t1 - t0))" -v kb="$rss_kb" -v out="$out" 'BEGIN {
    printf "requests %d, error responses %d, %.1f s, server RSS %.2f GiB -> %s\n",
        n, e, ns / 1e9, kb / 1048576, out
}'
