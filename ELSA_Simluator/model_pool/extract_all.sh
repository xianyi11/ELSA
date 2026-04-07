#!/usr/bin/env bash
# Unpack all archives in this directory into sibling folders (same base name as the archive).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

shopt -s nullglob

extracted=0

for f in *.zip; do
  dest="${f%.zip}"
  mkdir -p "$dest"
  echo "Unzipping: $f -> $dest/"
  unzip -o -q "$f" -d "$dest"
  extracted=$((extracted + 1))
done

for f in *.tar.gz *.tgz; do
  [[ -e "$f" ]] || continue
  dest="${f%.tar.gz}"
  dest="${dest%.tgz}"
  mkdir -p "$dest"
  echo "Extracting: $f -> $dest/"
  tar -xzf "$f" -C "$dest"
  extracted=$((extracted + 1))
done

for f in *.tar.bz2 *.tbz2; do
  [[ -e "$f" ]] || continue
  dest="${f%.tar.bz2}"
  dest="${dest%.tbz2}"
  mkdir -p "$dest"
  echo "Extracting: $f -> $dest/"
  tar -xjf "$f" -C "$dest"
  extracted=$((extracted + 1))
done

for f in *.tar; do
  [[ -e "$f" ]] || continue
  dest="${f%.tar}"
  mkdir -p "$dest"
  echo "Extracting: $f -> $dest/"
  tar -xf "$f" -C "$dest"
  extracted=$((extracted + 1))
done

if [[ "$extracted" -eq 0 ]]; then
  echo "No archive files (*.zip, *.tar.gz, *.tar.bz2, *.tar) found in $DIR"
  exit 0
fi

echo "Done. Extracted $extracted archive(s) under $DIR"
