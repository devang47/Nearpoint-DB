#!/usr/bin/env bash
# Fetch a texmex dataset into ${NEARPOINT_DATA:-~/data/Nearpoint}/<name>/.
# Usage: tools/fetch-dataset.sh siftsmall|sift1m
# Resumable: re-running continues a partial download and skips finished ones.
set -euo pipefail

name=${1:?usage: $0 siftsmall|sift1m}
# Each file's expected size = vectors * (4-byte dim + dim * 4 bytes).
case $name in
  siftsmall) archive=siftsmall.tar.gz extracted=siftsmall p=siftsmall  # ~5 MB, ~18 MB extracted
             files="base:10000:516 query:100:516 learn:25000:516 groundtruth:100:404" ;;
  sift1m)    archive=sift.tar.gz extracted=sift p=sift                 # ~161 MB, ~577 MB extracted
             files="base:1000000:516 query:10000:516 learn:100000:516 groundtruth:10000:404" ;;
  *) echo "unknown dataset: $name (want siftsmall or sift1m)" >&2; exit 2 ;;
esac

root=${NEARPOINT_DATA:-$HOME/data/Nearpoint}
dest=$root/$name
mkdir -p "$root"
df -h "$root" | tail -1

if [[ ! -d $dest ]]; then
  echo "downloading $archive (the texmex FTP mirror is slow, ~30 KB/s)"
  curl -fsSL -C - --retry 3 -o "$root/$archive" \
    "ftp://ftp.irisa.fr/local/texmex/corpus/$archive"
  tar -xzf "$root/$archive" -C "$root"
  [[ $extracted == "$name" ]] || mv "$root/$extracted" "$dest"
  rm "$root/$archive"
fi

for f in $files; do
  IFS=: read -r kind count record <<<"$f"
  file=$dest/${p}_$kind.$([[ $kind == groundtruth ]] && echo ivecs || echo fvecs)
  want=$((count * record))
  got=$(stat -f%z "$file")
  [[ $got == "$want" ]] || { echo "$file: $got bytes, expected $want" >&2; exit 1; }
done
echo "$name ok: $dest ($(du -sh "$dest" | cut -f1))"
