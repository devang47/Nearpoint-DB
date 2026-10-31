#!/bin/zsh
# Streams the BIGANN 1B base into a billion-mode build directory (Phase 9),
# resuming from the committed row after any failure, until `refine` is
# published. The directory must already hold `quant` (`billion train`).
# Usage: tools/billion-build.sh <dir> [total_rows]   (default 1,000,000,000)
set -u
dir=$1 total=${2:-1000000000}
bin=${BILLION_BIN:-./target/release/billion}
here=${0:A:h}
for attempt in {1..200}; do
  [[ -f "$dir/refine" ]] && { echo "stream complete"; exit 0; }
  from=$("$bin" status --dir "$dir") || exit 1
  echo "=== $(date '+%F %T') attempt $attempt from row $from"
  "$here/billion-stream.sh" "$from" "$total" | "$bin" build --dir "$dir" --from "$from" --rows "$total"
  [[ -f "$dir/refine" ]] || sleep 30
done
echo "gave up after 200 attempts" >&2
exit 1
