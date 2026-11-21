# Compare two runs of the same requests, response by response:
#   jq -n --slurpfile a ORACLE.jsonl --slurpfile b CANDIDATE.jsonl -f compare.jq
# Prints the counts of questions, of identical answers and of Choice
# decisions that differ, and the largest absolute difference of any reported
# number (probability, noul, score) and of any raw log-probability (debug).

def numbers:
  if .type == "noul" then [.noul]
  elif .type == "choice" then [.probabilities[]] + [.confidence]
  else [.probabilities[]] + [.score, .confidence] end;

def maxabs(xs): [xs | fabs] | (max // 0);

($a | length) as $n
| if $n != ($b | length) then error("the runs have \($n) and \($b | length) responses") else . end
| [ range(0; $n) as $i
    | $a[$i].answers | to_entries[]
    | { x: .value, y: $b[$i].answers[.key] } ] as $pairs
| [ range(0; $n) as $i
    | [$a[$i].debug[]?.raw_logprobs] as $la
    | [$b[$i].debug[]?.raw_logprobs] as $lb
    | range(0; $la | length) as $q
    | range(0; $la[$q] | length) as $o
    | $la[$q][$o] - $lb[$q][$o] ] as $lp
| {
    responses: $n,
    questions: ($pairs | length),
    identical_answers: ([$pairs[] | select(.x == .y)] | length),
    choice_decisions_differ: ([$pairs[] | select(.x.type == "choice" and .x.choice != .y.choice)] | length),
    max_reported_diff: maxabs($pairs[] | [(.x | numbers), (.y | numbers)] | transpose[] | .[0] - .[1]),
    max_raw_logprob_diff: maxabs($lp[]),
    raw_logprobs_compared: ($lp | length)
  }
