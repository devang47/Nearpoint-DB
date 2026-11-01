#!/bin/zsh
# Writes rows [first, total) of the BIGANN 1B base (u8, dim 128, no header)
# to stdout, for `billion build`. Fetches 64 MB range chunks,
# `par` at a time with the next group already downloading, and deletes each
# chunk once it is piped on: at most 2 * par chunks (~0.5 GB) of raw data
# are ever on disk. Exits non-zero if a chunk fails after curl's retries;
# the caller resumes from the build's committed row.
# Usage: tools/billion-stream.sh <first_row> <total_rows>
set -u
url=${BIGANN_URL:-https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin}
first=$1 total=$2 dim=128 chunk=500000 par=${STREAM_PAR:-4}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/np-stream.XXXXXX")
# Each fetch is a subshell running curl: stop the curls, not just the
# subshells, and clean up on signals too.
trap 'for p in ${(v)pid}; do pkill -P $p; kill $p; done 2>/dev/null; rm -rf "$tmp"' EXIT
trap 'exit 1' INT TERM HUP PIPE
n=$(( (total - first + chunk - 1) / chunk ))
typeset -A pid
fetch() {
  local a=$(( first + $1 * chunk )) b
  b=$(( a + chunk )); (( b > total )) && b=$total
  curl -fsS --retry 10 --retry-all-errors --retry-delay 5 \
    -r $(( 8 + a * dim ))-$(( 8 + b * dim - 1 )) -o "$tmp/$1.part" "$url" &&
    [[ $(stat -f%z "$tmp/$1.part") == $(( (b - a) * dim )) ]] &&
    mv "$tmp/$1.part" "$tmp/$1"
}
start() {
  local i p
  for (( i = $1; i < $1 + par && i < n; i++ )); do
    fetch $i &
    p=$! # zsh: `pid[$i]=$!` would store the literal text
    pid[$i]=$p
  done
}
start 0
for (( g = 0; g < n; g += par )); do
  start $(( g + par ))
  for (( i = g; i < g + par && i < n; i++ )); do
    wait $pid[$i] && [[ -f "$tmp/$i" ]] || { echo "chunk $i failed" >&2; exit 1; }
    cat "$tmp/$i" || exit 1
    rm -f "$tmp/$i"
    unset "pid[$i]"
  done
done
