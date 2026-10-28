#!/usr/bin/env bash
# Fetch a texmex dataset into ${NEARPOINT_DATA:-~/data/Nearpoint}/<name>/.
# Usage: tools/fetch-dataset.sh siftsmall|sift1m [--mirror hf]
#   default      canonical texmex FTP (ftp.irisa.fr), ~30 KB/s
#   --mirror hf  third-party Hugging Face copy of the extracted sift1m files
#                (huggingface.co/datasets/qbo-odp/sift1m). Sizes are checked
#                here; bench's brute-force run must reproduce its ground truth.
# Resumable: re-running continues a partial download and skips finished ones.
set -euo pipefail

name=${1:?usage: $0 siftsmall|sift1m [--mirror hf]}
mirror=${3:-ftp}
[[ ${2:-} == "" || ${2:-} == --mirror ]] || { echo "unknown option: $2" >&2; exit 2; }
# Each file's expected size = vectors * (4-byte dim + dim * elem_size).
case $name in
  siftsmall) archive=siftsmall.tar.gz extracted=siftsmall p=siftsmall  # ~5 MB, ~18 MB extracted
             files="base:10000:516 query:100:516 learn:25000:516 groundtruth:100:404" ;;
  sift1m)    archive=sift.tar.gz extracted=sift p=sift                 # ~161 MB, ~577 MB extracted
             files="base:1000000:516 query:10000:516 learn:100000:516 groundtruth:10000:404" ;;
  *) echo "unknown dataset: $name (want siftsmall or sift1m)" >&2; exit 2 ;;
esac
[[ $mirror == ftp || ($mirror == hf && $name == sift1m) ]] ||
  { echo "mirror $mirror not available for $name" >&2; exit 2; }

root=${NEARPOINT_DATA:-$HOME/data/Nearpoint}
dest=$root/$name
mkdir -p "$root"
df -h "$root" | tail -1

file_of() { echo "$dest/${p}_$1.$([[ $1 == groundtruth ]] && echo ivecs || echo fvecs)"; }

if [[ $mirror == hf ]]; then
  mkdir -p "$dest"
  for f in $files; do
    file=$(file_of "${f%%:*}")
    curl -fsSL -C - --retry 3 -o "$file" \
      "https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/$(basename "$file")"
  done
elif [[ ! -d $dest ]]; then
  echo "downloading $archive (the texmex FTP mirror is slow, ~30 KB/s)"
  curl -fsSL -C - --retry 3 -o "$root/$archive" \
    "ftp://ftp.irisa.fr/local/texmex/corpus/$archive"
  tar -xzf "$root/$archive" -C "$root"
  [[ $extracted == "$name" ]] || mv "$root/$extracted" "$dest"
  rm "$root/$archive"
fi

for f in $files; do
  IFS=: read -r kind count record <<<"$f"
  file=$(file_of "$kind")
  want=$((count * record))
  got=$(stat -f%z "$file")
  [[ $got == "$want" ]] || { echo "$file: $got bytes, expected $want" >&2; exit 1; }
done
echo "$name ok: $dest ($(du -sh "$dest" | cut -f1))"
