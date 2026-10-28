---
name: fetch-dataset
description: Download or prepare a benchmark dataset (SIFT, BIGANN, SPACEV, ground truth) into ~/data/Nearpoint/ under the Mac mini's disk limits. Use before any dataset download, before reading a dataset that isn't on disk yet, or when disk space is in question.
---

# Fetch a dataset safely

The internal SSD is 256 GB with no external storage. Datasets never go in
the repo.

1. **Check space:** `df -h ~` and `du -sh ~/data/Nearpoint 2>/dev/null`.
2. **State the size** of what you're about to download *and* its size
   after extraction.
3. **Over 5 GB (either number)? Ask the user first** and wait for a yes.
4. **Download** with `tools/fetch-dataset.sh` once it exists (task 1.3);
   until then, the same steps by hand: `curl -fL -C - -o <file> <url>`
   into `~/data/Nearpoint/<dataset>/`, then `tar -xzf`, then delete the
   archive.
5. **Prefixes, not whole files.** For a subset of a 1B file (e.g. BIGANN
   100M), fetch the 8-byte header plus `N * dim * elem_size` bytes with
   `curl -r 0-<end>`; loaders read the first N and ignore the header count.
6. **Never store raw 1B data.** Billion mode streams `curl` into the
   builder's stdin and keeps only the encoded output.
7. **Verify** file sizes against the format (count × dim × elem_size +
   header) and report what landed and where.
