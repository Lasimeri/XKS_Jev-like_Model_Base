#!/usr/bin/env bash
# typesafe.sh [BASE_URL] [API_KEY]: check a System One server against
# TypeSafe's documented contract (OpenAPI 0.2.0,
# https://docs.typesafe.ai/api/system-one; https://docs.typesafe.ai/models;
# read 2026-11-21) with raw curl requests: status codes, the 422 validation
# body ({"detail": [{loc, msg, type}]}), the 400 body ({"detail": ...}), the
# response shape, /v1/models. Default BASE_URL http://127.0.0.1:8097.
# With API_KEY, also checks the refusals without a key (403) and with a
# wrong one (401), as the hosted Jev gives them (measured 2026-11-21).
# Prints one PASS/FAIL line per case and a total; exits 1 on any failure.
set -uo pipefail
base=${1:-http://127.0.0.1:8097}
key=${2:-}
auth=()
[ -n "$key" ] && auth=(-H "Authorization: Bearer $key")
pass=0
fail=0

# check NAME EXPECTED_STATUS JQ_TEST METHOD PATH [BODY] [EXTRA CURL ARGS...]
check() {
    local name=$1 want=$2 test=$3 method=$4 path=$5 body=${6:-}
    shift 6 2>/dev/null || shift $#
    local out status json
    if [ -n "$body" ]; then
        out=$(curl -s -w '\n%{http_code}' -X "$method" "${auth[@]}" -H 'Content-Type: application/json' \
            "$@" --data-binary "$body" "$base$path")
    else
        out=$(curl -s -w '\n%{http_code}' -X "$method" "${auth[@]}" "$@" "$base$path")
    fi
    status=${out##*$'\n'}
    json=${out%$'\n'*}
    if [ "$status" = "$want" ] && jq -e "$test" >/dev/null 2>&1 <<<"$json"; then
        pass=$((pass + 1))
        printf 'PASS %-44s %s\n' "$name" "$status"
    else
        fail=$((fail + 1))
        printf 'FAIL %-44s got %s, want %s: %s\n' "$name" "$status" "$want" "${json:0:300}"
    fi
}

noul='{"model":"jev-latest","state":"I was charged twice. Please help.","questions":{"billing":{"type":"noul","instructions":"Is this message about billing?"}}}'
check 'models: documented list' 200 '(.models | map(.name) | index("jev-latest")) != null and (.models[0] | has("description") and has("release_date"))' GET /v1/models
check 'models: OpenAI-style data list kept' 200 '.data[0].id != null' GET /v1/models
check 'systemone: noul answer' 200 '.answers.billing.type == "noul" and (.answers.billing.noul | type) == "number" and (.usage.input_tokens | type) == "number" and (.usage.output_tokens | type) == "number" and (.model | type) == "string"' POST /v1/systemone "$noul"
check 'systemone: choice without instructions' 200 '.answers.t.type == "choice" and (.answers.t | has("choice") and has("confidence") and has("probabilities"))' POST /v1/systemone \
    '{"model":"jev-latest","state":"Thanks, all good now!","questions":{"t":{"type":"choice","criteria":{"angry":"An upset message","calm":null}}}}'
check 'systemone: one-level score' 200 '.answers.u.score == 0 and .answers.u.confidence == 1 and .answers.u.legend["0"] == "Can wait"' POST /v1/systemone \
    '{"model":"jev-latest","state":"x","questions":{"u":{"type":"score","criteria":["Can wait"]}}}'
check 'systemone: versioned jev id accepted' 200 '.answers.billing.type == "noul"' POST /v1/systemone "${noul/jev-latest/jev-1.14.0}"
check 'systemone: long-context header accepted' 200 '.answers.billing.type == "noul"' POST /v1/systemone "$noul" -H 'TypeSafe-Danger-Zone-Long-Context: true'
check '422: model missing' 422 '.detail[0].loc == ["body","model"] and .detail[0].type == "missing" and .detail[0].msg == "Field required"' POST /v1/systemone \
    '{"state":"s","questions":{"q":{"type":"noul"}}}'
check '422: no questions' 422 '.detail[0].type == "too_short"' POST /v1/systemone '{"model":"jev-latest","state":"s","questions":{}}'
check '422: score of 11 levels' 422 '.detail[0].loc == ["body","questions","u","score","criteria"] and .detail[0].type == "too_long"' POST /v1/systemone \
    '{"model":"jev-latest","state":"s","questions":{"u":{"type":"score","criteria":["0","1","2","3","4","5","6","7","8","9","10"]}}}'
check '422: unknown question type' 422 '.detail[0].type == "union_tag_invalid"' POST /v1/systemone \
    '{"model":"jev-latest","state":"s","questions":{"q":{"type":"rank"}}}'
check '422: not JSON' 422 '.detail[0].type == "json_invalid"' POST /v1/systemone '{"model": '
check '400: box question (text-only model)' 400 '(.detail | type) == "string"' POST /v1/systemone \
    '{"model":"jev-latest","state":"photo","questions":{"r":{"type":"box","instructions":"Locate the red rectangle."}}}'
check '400: image assets (text-only model)' 400 '(.detail | type) == "string"' POST /v1/systemone \
    '{"model":"jev-latest","state":"p","questions":{"q":{"type":"noul"}},"assets":{"p":{"type":"image","format":"png","data_base64":"AA=="}}}'
check '400: unknown model' 400 '(.detail | contains("unknown model"))' POST /v1/systemone "${noul/jev-latest/gpt-5}"
check '405: GET /v1/systemone' 405 '.detail == "Method Not Allowed"' GET /v1/systemone
check '404: unknown path' 404 '.detail == "Not Found"' GET /v2/nothing

if [ -n "$key" ]; then
    auth=()
    check '403: no key (as the hosted Jev)' 403 '.detail.error_type == "authentication_error"' GET /v1/models
    auth=(-H 'Authorization: Bearer wrong-key')
    check '401: wrong key (as the hosted Jev)' 401 '.detail.error_type == "authentication_error"' GET /v1/models
    auth=()
    check 'health stays open' 200 '.status == "ok"' GET /health
fi

echo "conformance: $pass passed, $fail failed against $base"
[ "$fail" -eq 0 ]
