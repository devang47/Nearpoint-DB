#!/usr/bin/env bash
# Fetch a dataset into ${NEARPOINT_DATA:-~/data/Nearpoint}/<name>/.
# Usage: tools/fetch-dataset.sh siftsmall|sift1m|bigann10m [--mirror hf]
#   default      canonical texmex FTP (ftp.irisa.fr), ~30 KB/s
#   --mirror hf  third-party Hugging Face copy of the extracted sift1m files
#                (huggingface.co/datasets/qbo-odp/sift1m). Sizes are checked
#                here; bench's brute-force run must reproduce its ground truth.
#   bigann10m    first 10M BIGANN vectors from big-ann-benchmarks (HTTPS,
#                u8bin): a byte-range prefix of the 1B base (~1.28 GB; its
#                header still says 1B), the 10k queries and the 10M ground
#                truth. texmex only has the base as .bvecs.gz, which can't be
#                range-fetched.
# Resumable: re-running continues a partial download and skips finished ones.
set -euo pipefail

name=${1:?usage: $0 siftsmall|sift1m|bigann10m [--mirror hf]}
mirror=${3:-ftp}
[[ ${2:-} == "" || ${2:-} == --mirror ]] || { echo "unknown option: $2" >&2; exit 2; }
# Each file's expected size = vectors * (4-byte dim + dim * elem_size).
case $name in
  siftsmall) archive=siftsmall.tar.gz extracted=siftsmall p=siftsmall  # ~5 MB, ~18 MB extracted
             files="base:10000:516 query:100:516 learn:25000:516 groundtruth:100:404" ;;
  sift1m)    archive=sift.tar.gz extracted=sift p=sift                 # ~161 MB, ~577 MB extracted
             files="base:1000000:516 query:10000:516 learn:100000:516 groundtruth:10000:404" ;;
  bigann10m) [[ $mirror == ftp ]] || { echo "bigann10m has no --mirror" >&2; exit 2; } ;;
  *) echo "unknown dataset: $name (want siftsmall, sift1m or bigann10m)" >&2; exit 2 ;;
esac
[[ $mirror == ftp || ($mirror == hf && $name == sift1m) ]] ||
  { echo "mirror $mirror not available for $name" >&2; exit 2; }

root=${NEARPOINT_DATA:-$HOME/data/Nearpoint}
dest=$root/$name
mkdir -p "$root"
df -h "$root" | tail -1

if [[ $name == bigann10m ]]; then
  url=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks
  # local name, bytes (u8bin/ibin: 8-byte header + payload), source
  set -- \
    base.10M.u8bin 1280000008 "$url/bigann/base.1B.u8bin" \
    query.public.10K.u8bin 1280008 "$url/bigann/query.public.10K.u8bin" \
    gt.10M.ibin 8000008 "$url/GT_10M/bigann-10M"
  mkdir -p "$dest"
  while (($#)); do
    file=$dest/$1 want=$2 src=$3
    shift 3
    have=$([[ -f $file ]] && stat -f%z "$file" || echo 0)
    # A range request on the first `want` bytes, resumed by appending from
    # `have`: this is how only the 10M prefix of the 128 GB base comes down.
    if ((have < want)); then
      # --max-filesize: abort if the server ignores the range and sends it all.
      curl -fsSL --retry 3 --max-filesize $((want - have)) \
        -r "$have-$((want - 1))" "$src" >>"$file"
    fi
    got=$(stat -f%z "$file")
    [[ $got == "$want" ]] || { echo "$file: $got bytes, expected $want" >&2; exit 1; }
  done
  echo "$name ok: $dest ($(du -sh "$dest" | cut -f1))"
  exit 0
fi

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
